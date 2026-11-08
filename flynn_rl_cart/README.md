# FLYNN 식 기반 MaleCNS 강화학습 카트

기존 TASS 실험에서 만든 `visual_motor_circuit`과 광수용체 hex 입력을 그대로 사용한다.
MuJoCo 2륜 카트가 빛 구체를 따라가도록 PPO로 학습한다.

## 신경식

논문의 식을 그대로 사용한다.

```text
h(t+1) = alpha*h(t) + (1-alpha)*tanh(W*h(t) + sensory(t) + bias)
```

- `h`: 약 887,000개 실행 노드의 상태
- `W`: 약 3,036만 개 원본 연결의 희소행렬
- `sensory`: 기존 TASS hex 매핑으로 샘플링한 약 5,896개 광수용체 입력
- `alpha`: superclass별 학습 가능한 leak
- `bias`: 노드별 학습 가능한 bias
- 출력: 주석이 `descending_neuron`인 상태를 작은 MLP가 좌우 바퀴 명령으로 변환

connectome에 없는 연결은 학습 중에도 생성하지 않는다.

## 연결 가중치 학습 모드

현재 회로의 연결별 파라미터는 3천만 개가 넘는다. 기본값은 현실적인 메모리 사용을 위해
`source_gain`이다.

```json
"weight_training": "source_gain"
```

각 presynaptic 노드의 gain을 학습해 해당 노드가 내보내는 원본 연결을 함께 조절한다. 구현은
`W @ (gain * h)` 형태라 희소행렬 자체의 거대한 밀집 gradient를 만들지 않는다. 전달물질 부호와
connectome topology는 보존한다.

논문처럼 모든 연결을 개별 학습하려면 다음으로 변경할 수 있다.

```json
"weight_training": "per_edge"
```

이 모드는 CSR backward 대신 edge별 기여를 `index_add`로 누적하지만, 가중치·gradient·Adam
상태만으로도 큰 GPU 메모리가 필요하며 truncated BPTT의 hidden state까지 포함하면 일반 PC에서
메모리 부족이 발생할 가능성이 높다. 연결을 고정하고
leak·bias·출력부만 학습하려면 `frozen`을 사용한다.

## 입력

카트의 좌우 MuJoCo 눈 렌더를 기존 `photoreceptor_hex_map.feather` 좌표에서 샘플링한다.
정책에는 구체 위치·거리·각도를 직접 전달하지 않는다.

구체 좌표와 카트 자세는 다음에만 사용한다.

- 보상 계산
- 학습 중 critic 입력
- 커리큘럼과 성공 판정

평가 시 actor는 광수용체→CNS→하행뉴런 상태만 사용한다.

## 커리큘럼

1. 에피소드 동안 정지한 구체 추적
2. 좌우 직선으로 움직이는 구체 추적
3. 경계 안에서 부드럽게 자유 이동하는 구체 추적

1단계에서도 구체 위치와 카트 시작 위치·방향은 에피소드마다 달라진다. 같은 출발 동작을 외워서
카메라 없이 성공하는 편법을 막기 위해서다.

최근 100회 성공률이 80% 이상이고 단계별 최소 에피소드 수를 채우면 자동 승급한다. 높은 단계에서도
20% 확률로 이전 단계를 다시 학습해 망각을 줄인다.

## 보상

- 구체와 거리 감소: 양의 보상
- 구체 도달: 큰 보상
- 움직이지 않음: 감점
- 제자리 회전: 감점
- 과도한 바퀴 출력: 작은 에너지 감점
- 벽 충돌: 감점
- 구체에서 멀어짐: 추가 감점
- 매 시간 단계: 작은 감점

움직임 여부는 바퀴 명령이 아니라 MuJoCo에서 측정한 실제 위치 변화로 판단한다.

## 설치

프로젝트 가상환경에 필요한 패키지를 사용자가 직접 설치한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\flynn_rl_cart\requirements.txt
```

RTX 4070에서는 공식 CUDA 13.2 빌드를 별도로 설치한다. 일반 PyPI의 CPU 전용
PyTorch가 다시 선택되지 않도록 CUDA 의존성을 별도 파일로 분리했다. 이미
`2.14.1+cpu`가 설치된 환경에서는 `--force-reinstall`이 필요하다.

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall -r .\flynn_rl_cart\requirements-cuda.txt
```

설치 확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## 학습

MuJoCo 화면을 띄우지 않는 headless 학습이다.

기본 설정은 `"device": "cuda"`이며 CUDA를 사용할 수 없을 때 CPU로 자동 전환하지 않고
오류를 표시한다. 시작 로그에 GPU 이름, GPU 메모리와 PyTorch CUDA 버전이 출력되어야 한다.

```powershell
.\.venv\Scripts\python.exe .\flynn_rl_cart\train.py
```

중단된 학습 재개:

```powershell
.\.venv\Scripts\python.exe .\flynn_rl_cart\train.py --resume .\flynn_rl_cart\checkpoints\latest.pt
```

checkpoint는 `checkpoints/`, JSONL 학습 기록은 `logs/training.jsonl`에 생성된다.

## 평가와 광수용체 시각화

```powershell
.\.venv\Scripts\python.exe .\flynn_rl_cart\evaluate.py --checkpoint .\flynn_rl_cart\checkpoints\latest.pt
```

두 창이 열린다.

- MuJoCo 창: 카트와 빛 구체를 외부에서 관찰
- 광수용체 창: actor의 `sensory` 배열에 실제로 입력된 좌우 광수용체 값

광수용체 창에서 원본 카메라 이미지는 보여주지 않는다. 각 점의 색이 해당 광수용체에 들어간
`0~1` 밝기 값이다.

조작:

- `Space`: 시작·일시정지
- `R`: 현재 단계 초기화
- `1`: 정지 구체
- `2`: 좌우 이동 구체
- `3`: 자유 이동 구체

## 주의

- 이 폴더의 코드는 작성만 했으며 설치, 학습, 테스트, 평가를 실행하지 않았다.
- 현재 회로는 매우 커서 `source_gain`도 GPU에서 느릴 수 있다.
- PyTorch sparse CSR의 CUDA 지원 여부는 설치된 PyTorch/CUDA 조합에 따라 다르다.
- reward가 상승해도 시각을 실제로 사용한다는 뜻은 아니다. 시작 위치 무작위화와 한쪽 눈 차단 등의
  평가가 필요하지만, 요청에 따라 별도 비교군 코드는 포함하지 않았다.
- 강화학습 결과는 checkpoint가 생성되기 전까지 존재하지 않는다.
