#!/usr/bin/env python3
"""기존 체크포인트에서 **가중치만** 이어받아 짧게 더 학습하는 config 를 만든다.

왜 --resume 로는 안 되나
  --resume 은 train_state 를 통째로 복원한다. 82k 짜리 런이 끝난 체크포인트는
  train_state.step 이 82000 이라 `range(start_step, num_train_steps)` 가 비고
  **한 스텝도 안 돈다.** 그리고 기존 config 들의 weight_loader 는 전부 pi05_base 를
  가리켜, 새 experiment 로 띄우면 81999 가 아니라 base 에서 시작한다.

무엇을 하나
  weight_loader 를 <ckpt>/params 로 바꾸고 optimizer·step 은 새로 초기화하는
  별도 config 를 등록한다. 짧은 warmup + decay 를 새로 건다. 통계는 그 체크포인트가
  쓰던 assets 를 그대로 가리켜 입력 분포가 비트 단위로 보존된다.

왜 AST 를 쓰나 (리뷰 N1)
  예전에는 `    TrainConfig(\\n        name="..."` 문자열로 블록을 찾고 다음
  `\\n    TrainConfig(` 직전까지를 복사했다. 두 가지가 깨졌다.
    · 이름 앞에 주석(`# <name>` marker)이 있으면 **앵커가 안 맞아 exit 1**
    · 복사 범위에 원본의 `# </name>` marker 까지 들어가, 붙인 자리에서
      `# </name>    TrainConfig(` 가 되어 **config.py 전체가 SyntaxError**
      그런데 생성기는 exit 0 으로 끝났다.
  이제 AST 로 TrainConfig 호출의 정확한 범위를 잡고, 쓰기 전에 ast.parse 로
  검증한 뒤 atomic write 한다.

사용법
  python3 prepare_posttrain_config.py <pi05> <원본 config 이름> <ckpt 경로> [스텝수]
"""
from __future__ import annotations
import ast, os, pathlib, re, sys

ASSET_ID = "RoboSynChallenge/cobotmagic_Sim_items_handover"


def _need_py310():
    """config.py 는 `match` 문을 쓴다. 3.10 미만에서는 ast.parse 가 실패한다.

    시스템 python3 이 3.8 인 컨테이너가 있다 (실제로 당했다). pi05 의 venv 로
    스스로 다시 실행한다. 그마저 없으면 조용히 넘어가지 않고 멈춘다.
    """
    if sys.version_info >= (3, 10):
        return
    if len(sys.argv) > 1:
        venv = pathlib.Path(sys.argv[1]) / ".venv/bin/python"
        if venv.exists() and not os.environ.get("_RSC_REEXEC"):
            os.environ["_RSC_REEXEC"] = "1"
            os.execv(str(venv), [str(venv), __file__, *sys.argv[1:]])
    sys.exit(f"★ Python {sys.version_info.major}.{sys.version_info.minor} 으로는 "
             f"config.py 를 파싱할 수 없다 (match 문). 3.10 이상으로 실행할 것 — "
             f"예: <pi05>/.venv/bin/python {pathlib.Path(__file__).name} ...")


def find_trainconfig(src: str, name: str) -> tuple[int, int]:
    """이름이 `name` 인 TrainConfig(...) 호출의 [시작, 끝) 바이트 범위.

    주석·marker 와 무관하게 호출 자체만 잡는다.
    """
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    offs, acc = [], 0
    for l in lines:
        offs.append(acc)
        acc += len(l)

    def pos(lineno, col):
        return offs[lineno - 1] + col

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Name) and f.id == "TrainConfig"):
            continue
        for kw in node.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant) \
                    and kw.value.value == name:
                return pos(node.lineno, node.col_offset), \
                       pos(node.end_lineno, node.end_col_offset)
    raise SystemExit(f"★ TrainConfig(name=\"{name}\") 를 못 찾았다")


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    _need_py310()
    pi05 = pathlib.Path(sys.argv[1])
    base = sys.argv[2]
    ckpt = pathlib.Path(sys.argv[3]).resolve()
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 12_000
    name = f"{base}_post{steps // 1000}k"

    params = ckpt / "params"
    if not params.exists():
        sys.exit(f"★ {params} 가 없다. 체크포인트 경로는 .../checkpoints/<step> 이어야 한다.")

    cfg = pi05 / "src/openpi/training/config.py"
    src = cfg.read_text()

    # ── 원본 호출을 정확히 떼어 온다 ───────────────────────────────────
    i, j = find_trainconfig(src, base)
    call = src[i:j]

    # ── 네 군데만 바꾼다 ───────────────────────────────────────────────
    new = call.replace(f'name="{base}"', f'name="{name}"', 1)

    wl = re.search(r'\n(\s*)weight_loader=weight_loaders\.CheckpointWeightLoader\([^)]*\),', new)
    if not wl:
        sys.exit("★ weight_loader 줄을 못 찾았다")
    ind = wl.group(1)
    new = new.replace(wl.group(0),
                      f'\n{ind}# RSC_POSTTRAIN: base 가 아니라 기존 체크포인트에서 가중치만 받는다.\n'
                      f'{ind}#   optimizer 와 step 은 새로 초기화된다 (--resume 과 다르다).\n'
                      f'{ind}weight_loader=weight_loaders.CheckpointWeightLoader(\n'
                      f'{ind}    "{params}"),', 1)

    # 통계: 그 체크포인트가 쓰던 것. config 이름이 바뀌면 assets_dirs 도 바뀐다.
    asset_line = (f'            assets=AssetsConfig(assets_dir="{ckpt / "assets"}",\n'
                  f'                                asset_id="{ASSET_ID}"),\n')
    ex = re.search(r' *assets=AssetsConfig\([^)]*\),\n', new)
    if ex:
        new = new.replace(ex.group(0), asset_line, 1)
    else:
        hook = re.search(r' *repo_id="[^"]*",\n', new)
        if not hook:
            sys.exit("★ repo_id 줄을 못 찾았다 — assets 를 끼울 자리가 없다")
        new = new.replace(hook.group(0), hook.group(0) + asset_line, 1)

    new = re.sub(r'(\n\s*)num_train_steps=[0-9_]+,', rf'\g<1>num_train_steps={steps:_},', new, count=1)
    new = re.sub(r'(\n(\s*))lr_schedule=_optimizer\.CosineDecaySchedule\([^)]*\),',
                 lambda m: f'{m.group(1)}lr_schedule=_optimizer.CosineDecaySchedule(\n'
                           f'{m.group(2)}    warmup_steps={max(200, steps // 20)},\n'
                           f'{m.group(2)}    peak_lr=1e-5,          # 82k 끝의 decay_lr(2.5e-6) 보다 조금 위\n'
                           f'{m.group(2)}    decay_steps={steps:_},\n'
                           f'{m.group(2)}    decay_lr=1e-6),', new, count=1)
    new = re.sub(r'(\n\s*)keep_period=[0-9_]+,', rf'\g<1>keep_period={max(1000, steps // 6):_},', new)

    for must in (f'name="{name}"', str(params), f"num_train_steps={steps:_}",
                 "assets=AssetsConfig(assets_dir="):
        if must not in new:
            sys.exit(f"★ 치환 실패: {must}")

    # ── 자기 marker 로 감싼다 (원본 marker 는 복사하지 않는다) ──────────
    b0, b1 = f"    # <{name}>\n", f"    # </{name}>\n"
    block = b0 + "    " + new + ",\n" + b1

    old_i = src.find(b0)
    if old_i >= 0:
        old_j = src.index(b1) + len(b1)
        if src[old_i:old_j] == block:
            print(f"  변경 없음: {name}")
            print("POSTTRAIN-CONFIG-DONE")
            return
        cand = src[:old_i] + block + src[old_j:]
        verb = "갱신"
    else:
        # 원본 호출이 시작하는 줄의 맨 앞. 단, 원본이 `# <base>` marker 로
        # 감싸여 있으면 그 **위**에 넣어야 한다 — marker 안에 넣으면
        # prepare_mistake_config 가 그 블록을 통째로 갈아끼울 때 같이 지워진다
        # (리뷰 N1-4 에서 실제로 사라졌다).
        line_start = src.rfind("\n", 0, i) + 1
        # 원본이 marker 쌍 안에 있으면 그 **쌍 전체의 위**로 올린다. 사이에 주석이
        # 끼어 있어도 상관없다 — 판정은 'marker 쌍이 호출을 감싸는가' 하나다.
        om, cm = f"    # <{base}>\n", f"    # </{base}>\n"
        k = src.rfind(om, 0, line_start)
        if k >= 0:
            end = src.find(cm, k)
            if end > i:                     # 닫는 marker 가 호출 뒤에 있다 = 감싸고 있다
                line_start = k
        cand = src[:line_start] + block + src[line_start:]
        verb = "추가"

    # ── 쓰기 전에 반드시 파싱한다 ──────────────────────────────────────
    try:
        ast.parse(cand)
    except SyntaxError as e:
        sys.exit(f"★ 생성 결과가 문법 오류다 ({e.lineno}행: {e.msg}) — 쓰지 않았다")
    tmp = cfg.with_suffix(".py.tmp")
    tmp.write_text(cand)
    tmp.replace(cfg)

    print(f"  {verb}: {name}")
    print(f"    가중치  {params}")
    print(f"    통계    {ckpt / 'assets'}")
    print(f"    스텝    {steps:,} (warmup {max(200, steps // 20):,}, peak_lr 1e-5 -> 1e-6)")
    print(f"    주의    --resume 을 쓰지 말 것. --overwrite 로 새 experiment 를 띄운다.")
    print("POSTTRAIN-CONFIG-DONE")


if __name__ == "__main__":
    main()
