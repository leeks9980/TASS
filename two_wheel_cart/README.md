# TASS 2륜 카트 시뮬레이션

파리 관절과 근육 모델 대신, 시각→다리 운동 회로의 좌우 집단 평균을 2륜 카트에 직접 연결한다.

```text
왼쪽 앞다리 fl 운동뉴런 68개 평균 × motor_gain → 왼쪽 바퀴
오른쪽 앞다리 fl 운동뉴런 67개 평균 × motor_gain → 오른쪽 바퀴
```

학습된 출력 계층, 좌우 신호 혼합, 조향 규칙은 넣지 않았다. 양쪽 모두 부호를 보존한 전체 활성
평균을 사용한다. 양수는 전진, 음수는 후진 방향이며 최종 바퀴 속도는
`-max_wheel_speed`부터 `+max_wheel_speed` 사이로 제한한다. `somaSide`가 실제 동측 다리를
나타낸다는 현재 모델의 가정을 그대로 사용한다.

현재 `config.json`의 `motor_subclasses`는 `["fl"]`로 설정되어 앞다리 운동뉴런만 읽는다.
`["fl", "ml", "hl"]`로 바꾸면 이전처럼 모든 다리 운동뉴런을 사용한다.

앞다리 상태 135개에서 좌우 바퀴로 가는 작은 선형 디코더는 다음 명령으로 학습한다.

```powershell
.\.venv\Scripts\python.exe .\two_wheel_cart\train_decoder.py
```

학습 중에는 MuJoCo 빛 구체 위치로 정답 바퀴 속도를 만들지만 CNS 가중치는 변경하지 않는다.
생성된 `front_leg_decoder.npz`가 있고 `decoder_enabled`가 `true`이면 `mujoco_main.py`가
자동으로 읽으며, 실행 중에는 목표 좌표가 아니라 광수용체→CNS→앞다리 운동뉴런 상태만
사용한다. 첫 실험 디코더는 움직이는
카트에서 출력 포화가 발생했기 때문에 `decoder_enabled`의 기본값은 `false`다. 결과를 직접
재현할 때만 `true`로 바꾼다.

## MuJoCo 카트 + 초파리 광수용체 시야

카트의 바퀴 접촉, 이동과 회전을 MuJoCo에서 함께 확인하려면 다음 Python 코드를 직접 실행한다.

```powershell
.\.venv\Scripts\python.exe .\two_wheel_cart\mujoco_main.py
```

같은 실행에서 두 개의 창이 열린다.

- MuJoCo 창은 움직이는 2륜 카트를 바깥에서 관찰한다.
- `초파리 광수용체 시야` 창은 원본 카메라 화면 대신 약 5,896개 광수용체가 실제로 받은
  입력을 좌·우 hex 모자이크로 표시한다.

MuJoCo 카트의 좌우 눈 렌더러는 모두 카트 정면을 향하고 시야각은 70도다. 두 렌더 결과는 사용자에게 직접
표시하지 않고 기존 `photoreceptor_hex_map.feather`의 좌표에서 샘플링하는 데만 사용한다.
표시되는 각 점의 밝기가 CNS의 `sensory` 배열에 들어간 값이다. Space 또는 패널 버튼으로
시작·일시정지하고 `R`로 초기화한다. 노란 빛 구체는 자동으로 움직이지 않으며 패널 버튼이나
키보드로 직접 이동한다. `W/S`는 전후, `A/D`는 좌우, `Q/E`는 위아래 이동이다. 기본 높이는
카트 눈높이에 가까운 0.25 m이며 한 번에 0.15 m씩 움직인다.

복안 hex 위상은 보존하지만 실제 초파리 시야각, 좌우 중첩, 회전 방향과 분광 감도는 아직
생리학적으로 보정되지 않았다.

카트 모드는 반응 지연을 줄이기 위해 20 ms마다 신경망을 갱신한다. 최대 12홉 경로의 단순
전파 지연은 약 0.24초이며, 영상 시간 필터는 이전 영상 30%와 새 영상 70%를 사용한다.

## 실행

프로젝트 루트에서 다음 Python 파일을 직접 실행한다. 별도 CMD 실행 파일은 없다.

```powershell
.\.venv\Scripts\python.exe .\two_wheel_cart\main.py
```

초기 회로 로딩에는 메모리와 시간이 필요하다. 창에서 화면을 클릭하면 빛 자극 위치가 이동한다.
`시작`, `일시정지`, `한 단계`, `초기화`를 사용할 수 있으며 모터 증폭 슬라이더로 작은 신호를
눈에 보이는 바퀴 속도로 확대할 수 있다.

창을 열지 않고 계산 경로를 확인하려면 다음과 같이 실행한다.

```powershell
.\.venv\Scripts\python.exe .\two_wheel_cart\main.py --headless-steps 10
```

설정은 같은 폴더의 `config.json`에서 바꾼다. 기본 신경 동역학은 루트의 `experiment.json`을 읽는다.

## 단위 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\two_wheel_cart -p "test_*.py" -v
```
