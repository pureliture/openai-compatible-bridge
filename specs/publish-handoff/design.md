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
# auto-deploy proof 20260928T040348Z

## Green closeout proof marker

수정된 `argo_release.py` Running-phase revision 레이스 완화와 GoCD 자동 3-stage 정의(tools `66d3e415…`)로 첫 자동 배포 성공 후, GHA/GoCD 세 stage 모두 `Passed`인 완전한 green 증거를 얻기 위한 재실증 push marker다.
# green closeout proof 20260928T042706Z

## Fixed-tools green proof marker

runner-side handoff와 GoCD release agent 모두 `23b9808…` revision으로 정렬하고, Running 단계의 Argo revision은 완료 시점의 exact SHA로만 검증하도록 수정한 뒤 수행하는 단일 green proof push marker다.
# fixed-tools green proof 20260928T045500Z

## Final automatic-deployment proof marker

GoCD가 아직 생성하지 않은 trailing automatic stage를 `approval_type: null`로 표현하는 관측 형식까지 runner가 수용한다. 실제 실행 또는 완료 stage에는 `success`만 허용하는 검증을 유지한 최종 green proof push marker다.
# final automatic-deployment proof 20260928T050900Z
- [2026-09-28T05:21:16+00:00] 최종 green proof를 위한 마커 커밋입니다.

- [2026-09-28T05:36:51+00:00] release-tools pin 정렬 후 final green proof marker

- [2026-09-28T05:46:46+00:00] history pagination fix 후 final green proof marker
