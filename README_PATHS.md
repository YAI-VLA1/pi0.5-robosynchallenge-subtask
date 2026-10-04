# 스크립트는 저장소 한 벌만 쓴다

`/root/rsc_recover/gh_subtask` 가 그 저장소다. orchestration 스크립트
(`setup/setup_train.sh`, `setup/apply_patches.sh`, `mem/apply_all.sh`) 는
저장소 구조를 가정하므로 **거기서 실행할 것**. `/root/rsc_recover` 바로 아래
`apply_*.py` 들은 개별 패치라 어디서 불러도 된다(저장소 사본과 같아야 한다).

    bash /root/rsc_recover/gh_subtask/setup/apply_patches.sh /workspace/rsc_ws/RoboSynChallenge --all
