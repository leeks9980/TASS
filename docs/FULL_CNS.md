# 전체 CNS + 공개 초파리 몸체 → 기록 → 필요한 회로 찾기

현재 단계는 PC에서 전체 연결망을 보존하고, 공개 초파리 몸체에서 시각 반응과 운동 출력을
기록하는 단계다. 원하는 운동신경을 사전에 좁히거나, 회피 규칙·보행 패턴·학습된 정책으로
동작을 만들지 않는다. 관찰 이후 필요한 회로의 후보를 찾는다.

## 실행

`run_mujoco.cmd` 또는 `scripts/run_full_cns.cmd`를 실행한다. 공개 몸체 자산은 준비해 두었다.
재설치 시 최초 실행에서 몸체 원본을 다운로드하고 XML을 만든다.
기존 `scripts/install_navigation.cmd`의 MuJoCo·NumPy·SciPy 의존성을 사용한다.
FlyGym 전체 패키지나 학습 환경은 설치하지 않는다.

두 창이 열린다: MuJoCo 몸체 화면과 한국어 신경·관절 표시 창.

- Space 또는 시작 버튼: 시작/일시정지.
- R: 몸체·신경·출력 활성 초기화. 정지 상태로 돌아간다.
- C: 외부 시점/입력 카메라.
- 1: 물체 없음, 2: 밝게, 3: 어둡게, 4: 왼쪽 이동, 5: 오른쪽 이동,
  6: 정면 접근, 7: 이동 물체.
- 조건 변경은 전체 상태를 초기화하고 정지한다. 다시 Space로 시작한다.

기본은 이동 물체 조건이고, 자극은 가상 0.5초부터 시작한다.
물체는 고정된 세계 좌표계에서 움직인다. 로봇 위치를 따라가거나 회피 명령을 만들지 않는다.
표시 창에는 42개 관절의 각도·속도·신경 토크, 각 근육 기능 채널의 활동·활성·힘,
몸체 위치·전진축 속도·경로·발 접촉이 나온다. 정확한 축은 metadata의 joint_axes_local에 있다.

같은 실행에서 무물체 기준을 비교하려면:

```powershell
.\scripts\run_full_cns.cmd --record-gb 20
```

1번을 선택해 약 1초의 **가상 시간** 동안 실행하고 Space로 멈춘다.
다른 조건을 선택하고 Space로 시작하면 같은 신경 업데이트 번호의 무물체 관절 각도와 비교한다.
비교값은 표시용이며 출력에 더하거나 빼지 않는다. 무물체 조건에서 초기화하면 기준을 지운다.

전체 상태 float32만으로 업데이트당 약 337 MiB가 필요하며 물리 기록도 추가된다.
기본 10 GiB는 대략 30개 업데이트, 가상 약 1.5초 규모다.
여러 조건을 한 실행에서 비교하려면 저장 한도를 늘려야 한다.
한도는 완료 프레임 저장 후 확인하므로 마지막 프레임 크기만큼 넘을 수 있다.
PC 계산·저장 시간이 가상 시간보다 길 수 있다. 콘솔과 표시 창에서 단계를 확인한다.

## 공개 몸체와 운동 출력의 경계

원본: [NeuroMechFly / FlyGym v1.2.1](https://github.com/NeLy-EPFL/flygym/tree/v1.2.1).
고정 commit은 `c7affce924cb1c6add16619adf83be5c6b223e89`다.
원본 DeepFly3D XML, 실제 형상 STL 39개, 정적인 tripod 초기 자세, Apache-2.0 라이선스를
`data/fly_body/upstream`에 보존했다. 파일 해시와 변경 내역은 `manifest.json`에 있다.
초기 자세를 보행 순서로 재생하지 않는다.

- 원본의 형상, 질량, 다리 관절 축, 몸체 연결 구조를 사용한다.
- 몸체 6자유도와 원본의 회전 자유도 87개를 보존한다. 다리 관절은 66개다.
  머리·더듬이의 3축 힌지 묶음 7개는 볼 관절 7개로 바꿨다. 회전 자유도는 각각 3개이며
  관절 객체 수는 87개에서 73개로 줄었다. 실제 해부학적 볼 관절이라는 주장은 아니다.
  초기 자세는 원래 힌지 순서의 회전을 몸체 쿼터니언에 반영한다.
  변환 목록과 원래 관절 이름·축·각도는 body manifest에 보존한다.
- 더듬이의 pitch가 −90°에 가까워진 실패 기록에서 Euler 회전축 특이점이 확인돼
  회전 표현을 바꿨다. 적분기는 회전에 따른 속도 미분도 반영하는 `implicit`을 사용한다.
- 다리당 몸통–기절 3축, 대퇴 2축, 경골 1축, 발목 1축에 토크 출력 장치를 둔다: 총 42개.
  발의 나머지 4개 관절도 물리 계산에 남는다.
- 대퇴 회전 등 근거 있는 출력 대응이 없는 관절에는 신경 토크를 넣지 않는다.
- 중력과 지면·물체 접촉을 사용한다. 다리 간 자기 충돌은 켜지 않았다.
- 초기 자세와 관성에 따라 산정한 수동 스프링·감쇠가 있다. `springdamper="0.02 1"`로
  기준 자세의 관성을 사용해 시정수 0.02초, 감쇠비 1에 맞춘다. 생리학적으로 보정한 값은 아니다.
  이전 일정 강성(다리 0.2, 기타 2)은 작은 더듬이 관성에 비해 과도할 수 있어 교체했다.
  생물학적으로 검증한 관절 제한이 없어서 인위적 각도 한계를 추가하지 않았다.
- 몸통·머리·날개 등도 물리 모델에 있지만 이번 신경 출력 장치는 다리만 연결한다.

**몸체가 실제 형상을 사용한다는 것과 신경→근육 연결이 생물학적으로 완성됐다는 것은 별개다.**
현재는 공개 근육 기능 이름을 읽어 토크로 바꾸는 명시적인 가정이 필요하다.
공개 [FlyGym 근육 몸체](https://neuromechfly.org/api_reference/flygym/compose/fly/musculoskeletal/)도
현재 왼쪽 앞다리 근육만 지원한다. 이를 여섯 다리에 그대로 복제해 실제 근육 모델이라고 부르지 않는다.

`config/fly_motor_mapping.json`에서 다음 정보를 수정할 수 있다.

- channels: 다리·근육 기능·주석 정규식, 또는 근거가 확보된 정확한 body_ids.
- moments: 채널이 작용할 관절과 부호 있는 모멘트암(mm).
- 기본 분류: promotor, remotor/abductor, anterior/posterior rotator, adductor,
  trochanter flexor/extensor, tibia flexor/extensor, tarsus depressor/levator.
- 익명 MN, 일부 accessory·long tendon·기타 기능은 근거 없는 축에 배정하지 않는다.
  없는 중간/뒷다리 promotor 주석을 앞다리 주석으로 대신하지 않는다.
- 좌우는 기본 세포체/근원 주석을 사용한다. 운동 축삭의 교차 투사를 복원한 것은 아니다.
  실제 표적 근거가 있으면 body_ids로 별도 지정할 수 있다.

출력식:

```text
raw = 해당 그룹의 신경 활동 평균
target = clip(raw / activity_scale, 0, 1)
activation: target에 접근하는 1차 반응, 시정수 tau
force_uN = max_force_uN * activation
torque_nNm = signed_moment_mm @ force_uN
```

기본 활동 기준 1e-4, 최대 힘 40 µN, 모멘트암 ±0.05 mm, 활성 시정수 0.1초는 가정이다.
모멘트암 부호와 다축 작용도 확인이 필요한 가정이다. 근육 길이·속도·힘의 Hill 모델은 아니다.
매 프레임 최대값으로 활동을 재정규화하지 않는다.

```powershell
.\scripts\run_full_cns.cmd --activity-scale 0.0001 --muscle-force-uN 40 --muscle-tau 0.1 --record-gb 20
```

원본 단위는 mm·g·s다. 모델 힘 1단위 = 1 µN, 토크 1단위 = 1 nN·m.
이전 관찰 장치의 N·m 수치를 그대로 사용하면 안 된다.
일반화 힘 배열의 자유 몸체 병진 성분은 µN, 회전·힌지 성분은 nN·m이다.
이전 고정 경골 장치는 `scripts/run_muscle_rig.cmd`, 설명은 `FULL_CNS_TIBIA_RIG.md`에 보존했다.

## 전체 신경 데이터의 범위

공식 [MaleCNS 다운로드](https://male-cns.janelia.org/download/)의 flat 표 7개를 보존한다:
주석, 뉴런 전달물질, segment 통계, 전체 연결, 시냅스 위치, 파트너, 시냅스별 전달물질 확률.
약 24.37 GB이며 원본 SHA256 목록은 `data/male-cns-v1.0/manifest.json`에 있다.

현재 구성은 graph ID 88,404,403개, 주석 211,577개, 연결 151,856,684개다.
미주석 ID 88,192,826개는 완전한 뉴런으로 확인된 것이 아니라 segment를 포함한다.
시각·중앙 뇌·하행·상행·VNC·다른 감각·운동과 반복 연결을 계산에 남긴다.
기능·hop·최소 시냅스 수로 잘라낸 모델을 기본 실행하지 않는다.

EM 볼륨·segmentation·skeleton·Neo4j도 공개돼 있지만 모두 다운로드한 상태는 아니다.
시냅스 위치·confidence·확률은 원본과 역추적에 이용할 수 있지만 현재 점 뉴런 계산에서
전기적 구획, 전도 지연, 수용체의 효과를 복원한 것은 아니다.

입력은 기존 단일 카메라 160×120 → L1/L2/L3 어댑터다. 좌우 영상·hex 좌표 대응과
L1/L2 시간 변화, L3 시간 평균은 가정이다. 좌표 누락은 해당 쪽 평균을 사용한다.
광수용체는 그래프에 보존하지만 직접 자극하지 않는다.
접촉과 관절 상태는 기록하며, 고유수용 감각 뉴런에 임의로 입력하지 않는다.

```text
h_next = leak*h + (1-leak)*max(0, tanh(gain*W*h + camera_input))
```

기본 leak 0.65, gain 0.95. W는 시냅스 수의 postsynaptic 입력 합 정규화다.
`--weight-mode raw`로 원본 수를 사용할 수 있지만 gain과 포화를 함께 확인해야 한다.
GABA −, ACh +, Glutamate 기본 +/옵션 −는 수용체가 없는 상태의 가정이다.
기타 전달물질·미주석 segment는 + 구조 전달로 처리한다.
도파민 예측이 자동 학습·보상·자율성을 부여하지 않는다. LLM·강화학습·보행 생성기는 없다.
신경 업데이트는 기본 가상 0.05초마다 1회이며 생물학적 시간 보정값은 아니다.

## 기록, 재생, 분석

새 폴더는 `simulation_logs/full_fly_날짜/`다.

- metadata.json: 전체 그래프, 입력·NT·출력 가정, body ID 대응, 모든 이름·관절 주소·단위.
- annotations.feather: 모든 주석과 graph_index.
- all_leg_motor_neurons.csv, unmapped_leg_motor_neurons.csv: 전체/출력 미대응 다리 MN.
- motor_mapping.json, body_assets, body_model.mjb, initial_qpos.npy, source: 실행 당시 몸체·매핑·코드.
- events.jsonl: 조건, 입력/출력 시각, 신경 tick, 몸체 자세·속도·경로, 발 접촉, 출력 채널·관절.
- frame_*.npz: **매 신경 업데이트의 모든 graph 상태**, 입력 RGB, 입력값,
  그 사이 **모든 물리 단계**의 qpos/qvel/qacc, 몸체 위치·자세, 활성·힘·제어,
  수동·구속 힘, 접촉 기하·좌표·힘, 물체 위치·조명, 수치 경고.

수치 오류가 발생하면 자동 초기화 대신 종료하고 `failure.json`, `failure_body.npz`,
`failure_neural_state.npy`에 경고·관절·직전 상태·실패 프레임의 부분 기록을 남긴다.
실패 기록은 정상 완료 프레임으로 분석하지 않는다. 전체 실패 신경 상태 1개만큼 추가 공간이 필요하다.
실행 파일은 시작 시 기존 원본으로 몸체 XML을 다시 작성하므로 수치 설정 변경이 반영된다.

신경 상태는 무압축·무손실이다. 드문 활동은 0이 아닌 ID/값을 저장하며 나머지는 정확히 0이다.
활동 ID가 전체의 1/3 이상이면 전체 float32 배열을 저장한다. 상위 뉴런만 기록하지 않는다.
카메라와 신경 상태는 운동 출력 적용 전, 몸체 이벤트는 이어지는 물리 구간 종료 후다.
physics_contact_offsets[i:i+2]로 물리 단계 i의 접촉 배열 구간을 찾는다.
`motor_commands`는 새 몸체에서는 42개 관절 토크이며 단위는 nNm다.

```powershell
.\scripts\replay_fly_recording.cmd simulation_logs\full_fly_실행폴더
.\scripts\analyze_fly_recording.cmd simulation_logs\full_fly_실행폴더
.\scripts\analyze_full_cns.cmd simulation_logs\full_fly_실행폴더 --case approach --reference neutral --top 2000
```

재생은 저장한 물리 상태를 표시하며 신경망 계산과 물리 적분을 다시 하지 않는다.
Space 시작/정지, N/P 한 물리 단계 이동, R 처음, C 카메라다.
화면 접촉력 재계산보다 원본 NPZ에 기록된 접촉력을 분석 기준으로 사용한다.

fly_analysis/report_ko.md, body_timeline.csv, joint_timeline.csv, foot_timeline.csv에 운동을 요약한다.
annotated_motion_candidates.csv는 모든 주석 뉴런의 활동과 전진축 속도·회전·접촉의 상관 후보다.
초기화 에피소드별로 중심을 맞추고 자극 후 프레임만 사용한다. 변동이 없으면 상관은 정의되지 않는다.
이는 인과관계 증명이 아니며 시간·몸체 운동·중력·출력 매핑의 영향을 함께 받는다.

analyze_full_cns는 미주석 segment까지 포함한 **전체 그래프**의 조건 간 평균 반응 순위를 낸다.
기본 집계 시작은 새 기록의 자극 시작 시각이다. neutral이 없으면 활동 크기 순위라고 표시한다.
`--export circuit_extract/from_recording`은 후보 간 원본 연결을 내보낸다.
중간 전달·피드백을 누락할 수 있으므로 검증된 최소 회로가 아니다.
`src/trace_full_cns.py 실행폴더 --frame N --body ID --synapses`로 입력 연결과 원본 시냅스를 역추적할 수 있다.

경로가 생겼다고 전진 보행이라고 판단하지 않는다. 지면 미끄러짐·넘어짐도 경로에 들어간다.
몸체 앞 방향 속도, 자세, 발 접촉과 관절 운동을 함께 확인해야 한다.
동작이 없더라도 신경 기능 부재를 단정할 수 없다. 출력 대응·힘·시간·감각 피드백의 가정이 남아 있다.

## 이번 작업의 실행 범위

공개 몸체 다운로드, 원본 해시 확인, XML 작성만 수행했다.
사용자 요청에 따라 테스트, MuJoCo 구동, 신경망 실행은 수행하지 않았다.
움직임과 수치 안정성은 사용자 실행 결과로 확인해야 한다.
