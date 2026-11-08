# 주석 기반 앞다리 운동뉴런 MuJoCo 리그

기존 카트 실험과 분리된 실험이다. MaleCNS의 앞다리 `vnc_motor + fl` 뉴런을 모두 평균내지
않고 `type`, `instance`, `mancType`, `cell_type`의 기능 단어로 분류한다.

분류하는 기능은 다음과 같다.

| 주석 그룹 | 단순 MuJoCo 관절 명령 |
|---|---|
| promotor/protractor | 앞으로 내밀기 |
| remotor/retractor | 뒤로 당기기 |
| abductor | 벌리기 |
| adductor | 모으기 |
| levator/elevator | 올리기 |
| depressor | 내리기 |
| flexor | 굽히기 |
| extensor | 펴기 |

반대 기능의 부호 포함 평균 차이를 관절 목표각으로 사용한다.

```text
protraction = gain × (promotor - remotor)
abduction   = gain × (abductor - adductor)
levation    = gain × (levator - depressor)
flexion     = gain × (flexor - extensor)
```

뉴런 이름에 `remotor/abductor`처럼 여러 기능이 적혀 있으면 복수 그룹에 포함한다. 기능 이름이
없는 운동뉴런은 CNS 계산에는 남지만 이 리그의 관절을 직접 구동하지 않는다.

## 실행

프로젝트 루트에서 다음 Python 파일을 직접 실행한다. CMD 파일은 없다.

```powershell
.\.venv\Scripts\python.exe .\annotated_front_leg_mujoco\main.py
```

이 명령은 자동으로 실행하지 않았다. 사용자가 직접 실행을 요청하거나 위 명령을 실행해야 한다.

## 조작

- `Space`: 시작·일시정지
- `B`: 현재 기능별 신경 평균을 기준선으로 저장
- `R`: CNS, 리그, 기준선과 빛 위치 초기화
- `W/S`: 빛 구체 전후 이동
- `A/D`: 빛 구체 좌우 이동
- `Q/E`: 빛 구체 상하 이동

먼저 정면의 일정한 장면에서 잠시 진행한 뒤 `B`를 누르면 지속 활성과 좌우 고정 편향을 기준선으로
제거할 수 있다. 이후 빛 구체를 움직이며 기능 그룹별 변화량과 관절 반응을 관찰한다.

## 해석 제한

- 문자열 주석으로 기능을 분류한 것이며 완전한 근육 대응표가 아니다.
- 세포체 좌우를 실제 지배 다리 좌우로 가정한다.
- 단순 4관절 리그의 축과 모멘트암은 생물학적으로 검증되지 않았다.
- 음수 신경 상태는 발화율이 아니라 현재 signed-tanh 계산 상태다.
- 이 실험은 걷기 성능을 평가하지 않고, 기능 주석별 신호가 구별되는지 관찰한다.
