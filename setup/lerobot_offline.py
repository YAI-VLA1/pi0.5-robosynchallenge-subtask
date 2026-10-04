"""로컬 데이터셋을 Hub 조회 없이 열게 한다.

왜 필요한가
  lerobot 0.1.0 의 LeRobotDataset / LeRobotDatasetMetadata 는 생성자에서
  `get_safe_version(repo_id, CODEBASE_VERSION)` 을 **무조건** 부른다. 그 함수는
  `api.list_repo_refs()` 로 Hub 에 붙어 데이터셋 repo 에 `v2.1` 같은 코드베이스
  버전 태그가 달려 있는지 확인한다.

  RoboSynChallenge/cobotmagic_Sim_click_bell 에는 그 태그가 없다. 남의 repo 라
  우리가 달 수도 없다. 그래서 파일이 전부 로컬에 있어도
  `RevisionNotFoundError: Your dataset must be tagged with a codebase version` 로 죽는다.
  HF_HUB_OFFLINE=1 은 해결이 아니다 — 같은 호출이 OfflineModeIsEnabled 로 바뀔 뿐이다.

무엇을 하는가
  get_safe_version 이 태그 이름("v2.1") 대신 **기본 브랜치("main")** 를 돌려주게 한다.
  버전을 그대로 돌려주면 이번엔 뒤에서 snapshot_download(revision="v2.1") 가
  404 를 낸다 — 존재하지 않는 태그이기 때문이다. "main" 은 항상 존재하므로
  Hub 를 거치는 경로도 깨지지 않는다.

  우리는 meta/info.json 의 codebase_version 이 v2.1 이고 로컬 파일이 완전하다는 것을
  이미 확인했으므로, 태그 검사는 우리 상황에서 정보를 주지 않는다.

  학습 경로에는 쓰지 않는다. 분석 스크립트 전용이다.

사용법
  import lerobot_offline; lerobot_offline.patch()
  # 그 다음에 openpi.training.data_loader 를 쓴다
"""
from __future__ import annotations


def patch() -> None:
    from lerobot.common.datasets import lerobot_dataset, utils

    def _default_branch(repo_id, version):  # noqa: ARG001
        return "main"

    utils.get_safe_version = _default_branch
    # lerobot_dataset 은 이름으로 import 해 두므로 그쪽 참조도 바꿔야 한다.
    if hasattr(lerobot_dataset, "get_safe_version"):
        lerobot_dataset.get_safe_version = _default_branch
