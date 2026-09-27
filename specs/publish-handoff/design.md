# 게시 후 내부 handoff job

`build-publish.yml`의 `handoff` job은 보호된 main push에서 `test/publish/verify`가
성공하고 repository variable `ATLAS_HANDOFF_ENABLED == 'true'`일 때만 실행한다.
기본 미설정 상태에서는 활성화되지 않는다. 이 변경에서 runner 등록·예약·배포는 하지 않았다.

- 전용 Linux self-hosted label: `atlas-publish-handoff`.
- 같은 실행의 `bridge-build-receipt`를 받고 publish 출력과 세 선행 job 결과를 전달한다.
- 이 job에는 Actions/contents/packages read 권한만 있으며 source checkout은 하지 않는다.
- `/opt/atlas-publish-handoff/`의 사전 설치·고정한 k3s-infra 도구를 isolated Python으로 실행한다.
- writer SSH key, GoCD token, GET 전용 kubeconfig는 내부 runner의 별도 read-only mount다.
  일반 GitHub-hosted runner나 repository workflow secret으로 옮기지 않는다.
- 실패한 handoff만 재실행하면 현재 execution attempt와 같은 run의 이전 publication attempt를
  각각 API로 검증한다. receipt attempt의 세 선행 job 성공 및 현재 outputs 일치가 필요하다.
  동일 릴리스는 영속 상태를 재사용하며, 불명확한 POST는 반복하지 않는다.

구현·설정 계약은 k3s-infra의 `docs/publish-handoff.md`와
`ci/source_delivery/{publish_contract,publish_handoff,handoff_runtime}.py`가 소유한다.
flag를 켜기 전에 새 도구 revision, 내부 인증, 영속 상태, GoCD 접수 전용 profile과
`handoff-receipt → production-approval(manual) → production-sync(manual)` 정의를 qualification해야 한다.
현재 두 stage 정의에 예약 호출만 붙여 두 manual gate가 보존된다고 가정하지 않는다.

이번 로컬 구현과 기존 hybrid handoff #5/#6 성공은 새 main push의 자동 연결 성공 증거가 아니다.
활성화 후 별도 검증에서도 실제 배포 승인은 사람이 수행한다.

## Live qualification marker

클러스터 전용 runner·GoCD 3-stage 정의·제한 예약 경로 qualification 이후 첫 자동 연결 실증을 위한 marker 커밋이다. 실제 배포 승인은 포함하지 않는다.
