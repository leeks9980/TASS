# TASS 폴더 안내

모든 실행 예시는 `D:\code\TASS`를 작업 폴더로 사용한다.

| 위치 | 내용 |
|---|---|
| `run_mujoco.cmd` | 기본 초파리 몸체 시뮬레이션 실행 입구 |
| `src/` | 데이터 수집·회로 구성·신경 모델·시뮬레이션·GUI·기록 분석 Python 코드 |
| `scripts/` | 실행, 데이터 다운로드, 설치, 재생, 분석용 CMD 및 PowerShell 스크립트 |
| `config/` | 운동신경에서 관절로 연결하는 `fly_motor_mapping.json` |
| `models/` | 이전 바퀴·경골 장치·제한 회로 실험의 MuJoCo XML |
| `web/` | 이전 시각 반응 GUI의 HTML |
| `docs/` | 전체 CNS 설명, 이전 경골 장치 설명, 폴더 안내와 파일 이동 목록 |
| `tests/` | 기존 회로 추출·간단 신경 모델 검증 코드 |
| `logs/` | 오류 로그 및 폴더 정리 전 수정 파일의 백업 |
| `data/` | 공개 데이터와 공개 초파리 몸체 자산, 데이터 목록 |
| `data/legacy/` | 기존 `male_cns_all_neurons.csv` |
| `full_cns/` | 전체 연결망의 계산용 그래프 |
| `circuit_extract/` | 추출된 회로 |
| `navigation_circuit/` | 이전 제한 회로 실험의 그래프 |
| `L2_downstream_trace/` | 기존 L2 하위 연결 수집 및 경로 분석 결과 |
| `simulation_logs/` | 실행별 신경 상태·몸체 상태·카메라·실행 당시 소스와 자산 |
| `.deps/` | 기존 Python 의존성 패키지 |

`requirements.txt`, `requirements-navigation.txt`, `README.md`는 프로젝트 루트에 있다.
대용량 데이터 폴더와 기존 실행 기록은 이동하지 않았다. 기존 기록에 저장된 데이터 경로와 원본 소스는 유지된다.
공개 초파리 몸체의 실제 실행 XML은 기존처럼 `data/fly_body/arena.xml`이다.

## 실행 및 분석

```powershell
.\run_mujoco.cmd
.\scripts\run_full_cns.cmd --record-gb 20
.\scripts\analyze_fly_recording.cmd simulation_logs\full_fly_실행폴더
.\scripts\replay_fly_recording.cmd simulation_logs\full_fly_실행폴더
python src/trace_full_cns.py simulation_logs/full_fly_실행폴더 --frame 1 --body 뉴런ID
python src/Restore_Path.py
```

경로를 변경한 뒤 시뮬레이션이나 기존 테스트를 실행하지 않았다. Python 문법, 이동 파일의 존재, 소스 간 import와 실행 스크립트의 참조 경로만 정적으로 확인한다.

주요 오류 기록은 `logs/full_cns_error.log`다. MuJoCo가 자체 생성하는 `MUJOCO_LOG.TXT`는 실행 작업 폴더에 다시 생길 수 있다.

## 이동 기록과 백업

[file_moves.json](file_moves.json)에 기존 위치·새 위치·이동 전 크기와 SHA-256을 기록했다.
[file_organization_changes.json](file_organization_changes.json)에 수정된 파일과 정리 전 백업 위치를 기록했다.
수정 전 파일 내용은 `logs/folder_organization_backup/`에 보관했다. 기존 시뮬레이션의 `source/` 사본은 수정하지 않았다.

현재 실행에 대한 설명은 [FULL_CNS.md](FULL_CNS.md), 이전 고정 경골 장치는 [FULL_CNS_TIBIA_RIG.md](FULL_CNS_TIBIA_RIG.md)를 참고한다.
