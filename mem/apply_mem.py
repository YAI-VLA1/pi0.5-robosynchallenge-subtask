#!/usr/bin/env python3
"""SigLIP ViT 에 MEM short-term video memory 를 넣는다 (JAX/Flax 이식).

근거
  MEM 논문 arXiv 2603.03596 §III-C / Appendix C, PI 연구 페이지,
  그리고 공개 구현(LeRobot pi05 memory.py)에서 확인된 합성 방식.
  PI 공식 코드는 공개돼 있지 않다 — 이건 이식이지 재현 주장이 아니다.

합성 (중요)
  공간 블록과 시간 블록을 따로 쌓지 않는다. Q/K/V 를 **한 세트**만 뽑고,
  시간 어텐션으로 V 를 섞은 뒤 그 V 로 공간 어텐션을 하고, W_O 를 **한 번만** 건다.

    U        = LN(Z) + e(t)                 (e 는 temporal 층의 QKV 입력에만)
    Q,K,V    = W_Q(U), W_K(U), W_V(U)
    V_time   = causal_softmax_t(Q·Kᵀ) V     (같은 패치의 과거만)
    Y        = softmax_p(Q·Kᵀ) V_time       (프레임 내부)
    Z        = Z + W_O(Y) ; Z = Z + MLP(LN2(Z))

  이러면 T=1 에서 시간 softmax 가 identity 라 기존 어텐션으로 **정확히** 환원된다.
  (공간→시간 두 블록으로 만들면 W_O 가 두 번 걸려 T=1 에서도 값이 달라진다.)

체크포인트 호환
  파라미터 경로를 그대로 재현한다: MultiHeadDotProductAttention_0/{query,key,value,out}.
  새 학습 파라미터는 0 개다 (시간 임베딩은 고정 sinusoidal).

과거 토큰 버리기
  마지막 temporal 층(0-based 23) 이후에는 시간 혼합이 없고 공간 어텐션은 프레임
  내부에서만 일어나므로, **현재 프레임 출력은 언제 버리든 동일하다**. 구현은
  인코더를 나온 뒤 버린다 — 느릴 뿐 수치는 같다.
"""
import pathlib, sys

F = "src/openpi/models/siglip.py"

A_IMPORT = "class MlpBlock(nn.Module):"
N_IMPORT = '''def temporal_pos_emb(num_frames: int, dim: int, dtype) -> jnp.ndarray:
    """고정 sinusoidal 시간 임베딩. 상대 인덱스 [-(T-1) .. 0], 현재가 0.

    sin(0)=0 이고 cos(0)-1=0 이므로 **e(0)=0** 이 정확히 보장된다 (논문 요구사항).
    학습 파라미터가 아니다.
    """
    t = jnp.arange(-(num_frames - 1), 1, dtype=jnp.float32)[:, None]   # oldest -> current
    half = dim // 2
    i = jnp.arange(half, dtype=jnp.float32)[None, :]
    w = jnp.exp(-jnp.log(10000.0) * (2.0 * i) / dim)
    e = jnp.concatenate([jnp.sin(t * w), jnp.cos(t * w) - 1.0], axis=-1)
    if e.shape[-1] < dim:                       # dim 이 홀수면 0 으로 채운다
        e = jnp.pad(e, ((0, 0), (0, dim - e.shape[-1])))
    return e.astype(dtype)                      # (T, dim)


class SpaceTimeAttention(nn.Module):
    """MEM 의 space-time separable attention.

    num_frames == 1 이면 기존 MultiHeadDotProductAttention 과 **수치적으로 동일**하다.
    파라미터 이름이 같아 기존 체크포인트를 그대로 읽는다.
    """

    num_heads: int = 12
    dtype_mm: str = "float32"
    num_frames: int = 1

    @nn.compact
    def __call__(self, x, is_temporal=None, frame_valid=None, deterministic=True):  # noqa: FBT002
        # x: (B*T, N, D)  — T 는 같은 카메라의 프레임 수, N 은 패치 수
        feat = x.shape[-1]
        head_dim = feat // self.num_heads
        dense = functools.partial(
            nn.DenseGeneral,
            features=(self.num_heads, head_dim),
            axis=-1,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=self.dtype_mm,
        )
        q = dense(name="query")(x)
        k = dense(name="key")(x)
        v = dense(name="value")(x)

        T = self.num_frames
        if T > 1:
            BT, N, H, Dh = q.shape
            B = BT // T

            def to_time(z):                      # (B*T,N,H,Dh) -> (B*N,T,H,Dh)
                return z.reshape(B, T, N, H, Dh).transpose(0, 2, 1, 3, 4).reshape(B * N, T, H, Dh)

            def to_space(z):                     # 되돌리기
                return z.reshape(B, N, T, H, Dh).transpose(0, 2, 1, 3, 4).reshape(BT, N, H, Dh)

            causal = jnp.tril(jnp.ones((T, T), dtype=bool))
            if frame_valid is not None:          # (B,T) True = 실제 관측
                keep = frame_valid.astype(bool)[:, None, :].repeat(N, axis=1).reshape(B * N, T)
                m = causal[None] & keep[:, None, :]
                # 전부 막히면 softmax 가 NaN 이 된다 — 자기 자신은 항상 열어 둔다
                m = m | jnp.eye(T, dtype=bool)[None]
                mask = m[:, None]                # (B*N,1,T,T)
            else:
                mask = causal[None, None]
            def _mix(args):
                qq, kk, vv = args
                return to_space(
                    nn.dot_product_attention(to_time(qq), to_time(kk), to_time(vv),
                                             mask=mask, dtype=self.dtype_mm,
                                             deterministic=deterministic))

            if is_temporal is not None:
                # lax.cond 라야 **실제로 건너뛴다**. jnp.where 는 양쪽을 다 계산한 뒤
                # 고르므로 27층 전부에서 시간 어텐션이 돌고 21층 분이 버려진다
                # (필요량의 4.5배). 분기는 이미 구한 q,k,v 위에서만 돌아 새 파라미터가 없다.
                v = jax.lax.cond(jnp.asarray(is_temporal, bool),
                                 _mix, lambda a: a[2], (q, k, v))
            else:
                v = _mix((q, k, v))

        y = nn.dot_product_attention(q, k, v, dtype=self.dtype_mm, deterministic=deterministic)
        return nn.DenseGeneral(
            features=feat, axis=(-2, -1),
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=self.dtype_mm, name="out",
        )(y)


class MlpBlock(nn.Module):'''

A_BLOCK = '''    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)'''
N_BLOCK = '''    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    num_frames: int = 1          # >1 이면 MEM 시간 경로가 생긴다

    @nn.compact
    def __call__(self, x, deterministic=True, is_temporal=None, frame_valid=None):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        if self.num_frames > 1:
            # 시간 임베딩은 temporal 층의 QKV 입력에만 더한다. residual 기준은 x 다
            # (x + e 를 층마다 누적하지 않는다).
            BT, N, D = y.shape
            e = temporal_pos_emb(self.num_frames, D, y.dtype)        # (T,D)
            e = jnp.tile(e, (BT // self.num_frames, 1))[:, None, :]  # (B*T,1,D)
            gate = jnp.asarray(is_temporal, y.dtype) if is_temporal is not None else 1.0
            y = y + gate * e
        y = out["sa"] = SpaceTimeAttention(
            num_heads=self.num_heads,
            dtype_mm=self.dtype_mm,
            num_frames=self.num_frames,
            name="MultiHeadDotProductAttention_0",
        )(y, is_temporal=is_temporal, frame_valid=frame_valid, deterministic=deterministic)'''


# ── Encoder: 층별 플래그를 scan 에 넘긴다 ───────────────────────────────
# 27개 층이 scan 으로 말려 있어 (체크포인트가 encoderblock/... (27,...) 레이아웃)
# 루프 본문이 '몇 번째 층인지' 를 모른다. 27칸짜리 불리언을 scanned input 으로
# 넘겨 반복마다 한 칸씩 받게 한다.
A_ENC = """    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:"""
N_ENC = """    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    num_frames: int = 1          # MEM: 1 이면 기존과 완전히 동일한 경로
    spacetime_stride: int = 4    # 매 4번째 층에 시간 어텐션 (논문값)

    def _temporal_flags(self):
        \"\"\"층별 시간 어텐션 여부. depth=27, stride=4 -> 0-based 3/7/11/15/19/23.\"\"\"
        i = jnp.arange(self.depth)
        return ((i + 1) % self.spacetime_stride) == 0

    @nn.compact
    def __call__(self, x, deterministic=True, frame_valid=None):  # noqa: FBT002
        out = {}
        mem = self.num_frames > 1

        if self.scan:"""

A_SCAN = """            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)"""
N_SCAN = """            scanned = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                # deterministic 공통, is_temporal 은 층마다 한 칸, frame_valid 공통
                in_axes=(nn.broadcast, 0, nn.broadcast) if mem else nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                num_frames=self.num_frames,
            )
            if mem:
                x, scan_out = scanned(x, deterministic, self._temporal_flags(), frame_valid)
            else:
                x, scan_out = scanned(x, deterministic)"""

A_LOOP = """                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)"""
N_LOOP = """                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    num_frames=self.num_frames,
                )
                if mem:
                    x, out[f"block{lyr:02d}"] = block_cur(
                        x, deterministic,
                        jnp.asarray((lyr + 1) % self.spacetime_stride == 0), frame_valid)
                else:
                    x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)"""


# ── _Module: (B,T,H,W,3) 를 받아 현재 프레임 토큰만 돌려준다 ───────────
# 프레임마다 패치 추출/공간 posemb 는 그대로 (프레임 내부 처리라 바꿀 게 없다).
# 인코더를 나온 뒤 과거 시점 토큰을 버려 **단일 프레임과 같은 토큰 수**를 유지한다.
A_VM = '    posemb: str = "learn"  # Can also be "sincos2d"'
N_VM = (
    '    posemb: str = "learn"  # Can also be "sincos2d"\n'
    '    num_frames: int = 1   # MEM: >1 이면 image 가 (B,T,H,W,3)'
)

A_VCALL = (
    '    def __call__(self, image, *, train=False):\n'
    '        out = {}\n'
    '\n'
    '        # Kevin edit: do patch extraction and posemb in float32,\n'
    '        # because I feel like it\'s a bit safer.\n'
    '        image = jnp.asarray(image, jnp.float32)'
)
N_VCALL = (
    '    def __call__(self, image, *, train=False, frame_valid=None):\n'
    '        out = {}\n'
    '\n'
    '        image = jnp.asarray(image, jnp.float32)\n'
    '\n'
    '        # MEM: (B,T,H,W,3) 이면 프레임을 배치로 펴서 패치를 뽑는다.\n'
    '        T = self.num_frames\n'
    '        if image.ndim == 5:\n'
    '            Bv, Tv = image.shape[:2]\n'
    '            if Tv != T:\n'
    '                raise ValueError(f"image 의 T={Tv} 가 num_frames={T} 와 다르다")\n'
    '            image = image.reshape(Bv * Tv, *image.shape[2:])\n'
    '        elif T > 1:\n'
    '            raise ValueError(f"num_frames={T} 인데 image 가 4차원이다")'
)

A_VENC = (
    '        x, out["encoder"] = Encoder(\n'
    '            depth=self.depth,\n'
    '            mlp_dim=self.mlp_dim,\n'
    '            num_heads=self.num_heads,\n'
    '            dropout=self.dropout,\n'
    '            scan=self.scan,\n'
    '            remat_policy=self.remat_policy,\n'
    '            dtype_mm=self.dtype_mm,\n'
    '            name="Transformer",\n'
    '        )(x, deterministic=not train)\n'
    '        encoded = out["encoded"] = x'
)
N_VENC = (
    '        x, out["encoder"] = Encoder(\n'
    '            depth=self.depth,\n'
    '            mlp_dim=self.mlp_dim,\n'
    '            num_heads=self.num_heads,\n'
    '            dropout=self.dropout,\n'
    '            scan=self.scan,\n'
    '            remat_policy=self.remat_policy,\n'
    '            dtype_mm=self.dtype_mm,\n'
    '            num_frames=T,\n'
    '            name="Transformer",\n'
    '        )(x, deterministic=not train, frame_valid=frame_valid)\n'
    '        if T > 1:\n'
    '            # 과거 시점 토큰을 버린다 — 마지막 temporal 층 이후로는 시간 혼합이\n'
    '            # 없고 공간 어텐션은 프레임 내부라 현재 출력은 바뀌지 않는다.\n'
    '            BT, Np, Dp = x.shape\n'
    '            x = x.reshape(BT // T, T, Np, Dp)[:, -1]\n'
    '        encoded = out["encoded"] = x'
)


# 과거 토큰을 버리면 배치가 B*T -> B 로 줄어드는데, x_2d reshape 가 옛 n 을 쓴다.
# (T=3 이면 마지막 차원이 1152 -> 384 로 뭉개져 head 에서 터진다.)
A_V2D = '        x_2d = jnp.reshape(encoded, [n, h, w, -1])'
N_V2D = (
    '        # encoded 의 배치는 과거 토큰을 버린 뒤 바뀔 수 있다 (B*T -> B).\n'
    '        # 캡처해 둔 n 을 쓰면 채널이 뭉개진다.\n'
    '        x_2d = jnp.reshape(encoded, [encoded.shape[0], h, w, -1])'
)

# ── 데이터→모델 배선: (T,C,H,W) 통과 + frame_valid ───────────────────
# _parse_image 는 (C,H,W) 만 알고 있어 히스토리가 오면 (T,C,H,W) 를 그대로 흘린다.
A_PARSE = (
    'def _parse_image(image) -> np.ndarray:\n'
    '    image = np.asarray(image)\n'
    '    if np.issubdtype(image.dtype, np.floating):\n'
    '        image = (255 * image).astype(np.uint8)\n'
    '    if image.shape[0] == 3:\n'
    '        image = einops.rearrange(image, "c h w -> h w c")\n'
    '    return image'
)
N_PARSE = (
    'def _parse_image(image) -> np.ndarray:\n'
    '    image = np.asarray(image)\n'
    '    if np.issubdtype(image.dtype, np.floating):\n'
    '        image = (255 * image).astype(np.uint8)\n'
    '    if image.ndim == 4 and image.shape[1] == 3:\n'
    '        # MEM 히스토리: (T,C,H,W) -> (T,H,W,C). resize_with_pad 는 *b h w c 를\n'
    '        # 받으므로 프레임마다 독립적으로 리사이즈된다.\n'
    '        image = einops.rearrange(image, "t c h w -> t h w c")\n'
    '    elif image.shape[0] == 3:\n'
    '        image = einops.rearrange(image, "c h w -> h w c")\n'
    '    return image'
)

# Observation 에 frame_valid 를 단다 (기본 None = 기존과 동일)
A_OBS = '    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None'
N_OBS = (
    '    # MEM: 프레임별 유효 여부 (카메라별 (*b, T)). None 이면 히스토리 없음.\n'
    '    frame_valid: dict[str, at.Bool[ArrayT, "*b t"]] | None = None\n'
    '    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None'
)

# ── repack 이 _is_pad 를 통과시키게 (MEM 일 때만) ────────────────────
A_RP = (
    '                        "observation/state": self.state_key,\n'
    '                        "actions": "action",'
)
N_RP = (
    '                        "observation/state": self.state_key,\n'
    '                        "actions": "action",\n'
    '                        # MEM: LeRobot 이 주는 프레임 유효 마스크. repack 은 매핑에\n'
    '                        # 없는 키를 버리므로 여기서 명시해야 frame_valid 가 살아 간다.\n'
    '                        **({"observation/image_is_pad": self.image_key_high + "_is_pad",\n'
    '                            "observation/left_wrist_image_is_pad": self.image_key_left + "_is_pad",\n'
    '                            "observation/right_wrist_image_is_pad": self.image_key_right + "_is_pad"}\n'
    '                           if __import__("os").environ.get("PI05_MEM_FRAMES", "1") != "1" else {}),'
)

# ── EmbodiChainInputs 가 frame_valid 를 만든다 ───────────────────────
A_IN = (
    '            "image_mask": {'
)
N_IN = (
    '            # MEM: is_pad(True=패딩) 를 뒤집어 frame_valid 로 만든다.\n'
    '            **({"frame_valid": {\n'
    '                "base_0_rgb": ~np.asarray(data["observation/image_is_pad"]),\n'
    '                "left_wrist_0_rgb": ~np.asarray(data["observation/left_wrist_image_is_pad"]),\n'
    '                "right_wrist_0_rgb": ~np.asarray(data["observation/right_wrist_image_is_pad"]),\n'
    '            }} if "observation/image_is_pad" in data else {}),\n'
    '            "image_mask": {'
)

# ── embed_prefix 가 frame_valid 를 비전 타워에 넘긴다 ────────────────
A_EP = '            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)'
N_EP = (
    '            _fv = None if obs.frame_valid is None else obs.frame_valid.get(name)\n'
    '            image_tokens, _ = (\n'
    '                self.PaliGemma.img(obs.images[name], train=False)\n'
    '                if _fv is None\n'
    '                else self.PaliGemma.img(obs.images[name], train=False, frame_valid=_fv)\n'
    '            )'
)

def _patch(root, rel, pairs, marker):
    q = root / rel
    t = q.read_text()
    if marker in t:
        print(f"  {rel}: 이미 적용됨"); return
    for a, _ in pairs:
        if t.count(a) != 1:
            sys.exit(f"  {rel}: 앵커가 {t.count(a)}개 — 중단\n{a[:70]}")
    for a, n in pairs:
        t = t.replace(a, n, 1)
    q.write_text(t); print(f"  {rel}: 적용")


def main():
    root = pathlib.Path(sys.argv[1])
    _patch(root, "src/openpi/policies/libero_policy.py", [(A_PARSE, N_PARSE)], "MEM 히스토리: (T,C,H,W)")
    _patch(root, "src/openpi/models/model.py", [(A_OBS, N_OBS)], "frame_valid")
    _patch(root, "src/openpi/models/pi0.py", [(A_EP, N_EP)], "obs.frame_valid")
    p = pathlib.Path(sys.argv[1]) / F
    s = p.read_text()
    if "SpaceTimeAttention" in s:
        print("이미 적용됨"); return
    if "import functools" not in s:
        s = s.replace("import jax\n", "import functools\n\nimport jax\n", 1)
    pairs = ((A_IMPORT, N_IMPORT), (A_BLOCK, N_BLOCK),
             (A_ENC, N_ENC), (A_SCAN, N_SCAN), (A_LOOP, N_LOOP),
             (A_VM, N_VM), (A_VCALL, N_VCALL), (A_VENC, N_VENC),
             (A_V2D, N_V2D))
    for a, _ in pairs:
        if s.count(a) != 1:
            sys.exit(f"앵커가 {s.count(a)}개 — 중단:\n{a[:70]}")
    for a, n in pairs:
        s = s.replace(a, n, 1)
    p.write_text(s)
    print(f"패치 완료 {p}")


if __name__ == "__main__":
    main()
