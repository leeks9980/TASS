"""User-run GUI experiment. No automatic training, hidden obstacle oracle or LLM."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit('MuJoCo가 없습니다. install_navigation.cmd를 먼저 실행하세요.') from exc
import numpy as np
from navigation_model import NavigationCircuit, raw_command, visual_baseline


def run(args):
    if not (args.circuit / 'weights.npz').exists():
        raise SystemExit('먼저 prepare_navigation_circuit.py를 실행하세요. run_mujoco.cmd는 자동 준비합니다.')
    print('반복 연결 신경망 읽는 중…', flush=True)
    circuit = NavigationCircuit(args.circuit, args.nt)
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'models' / 'navigation_arena.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=120, width=160)
    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
    obstacles = {i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith(('obstacle_', 'wall_'))}
    # Viewer callbacks only enqueue requests; the simulation thread owns mutations.
    requests = []
    request_lock = threading.Lock()
    def key_callback(key):
        with request_lock:
            requests.append(key)
    output = ROOT / 'simulation_logs' / ('mujoco_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        'mode_at_start': args.mode, 'nt_mode': args.nt, 'camera': 'one 160x120 RGB camera',
        'input_types': ['L1', 'L2', 'L3'], 'trained': False,
        'spatial_mapping': 'normalized hex-column proxy per side; one image split into two fields',
        'missing_L3_coordinates': 'side-specific mean temporal luminance broadcast, not spatial mapping',
        'input_mapping_counts': circuit.input_report,
        'readout': 'bilateral mean descending activity; imposed exploration speed; untrained turn decoder',
        'body': 'planar two-wheel proxy, not a validated two-leg mechanism',
        'control_inputs': 'rendered camera only; contacts and pose used only for evaluation',
        'neuron_equation': 'h = 0.65*h + 0.35*tanh(0.95*W*h + input)',
        'nt_assumptions': 'structural: all positive; ach-gaba: ACh positive, GABA negative, all others zero',
        'circuit_report': json.loads((args.circuit / 'report.json').read_text(encoding='utf-8')),
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    paused, mode, first_person = True, args.mode, False
    episode = 0
    distance = 0.0
    collision_events = 0
    previous_collision = False
    previous_position = data.xpos[robot_id, :2].copy()
    last_print = 0.0
    print('GUI 준비 완료. Space: 시작/일시정지, R: 리셋, B: 회로/영상규칙 비교, C: 카메라 전환', flush=True)
    print('기본 RAW 모드는 학습되지 않았습니다. 장애물 회피 성공을 보장하지 않습니다.', flush=True)
    print(f'기록 폴더: {output}', flush=True)
    columns = ['episode', 'mode', 'sim_seconds', 'x', 'y', 'path_m', 'contact', 'collision_events',
               'dn_left', 'dn_right', 'linear_command', 'turn_command', 'left_motor', 'right_motor', 'compute_ms']
    try:
        with (output / 'frames.csv').open('w', newline='', encoding='utf-8-sig') as log, mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            writer = csv.DictWriter(log, fieldnames=columns)
            writer.writeheader()
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = robot_id
                viewer.cam.distance = 3
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -35
            while viewer.is_running():
                start = time.perf_counter()
                with request_lock:
                    keys = requests[:]
                    requests.clear()
                for key in keys:
                    if key == 32:
                        paused = not paused
                    elif key in [ord('R'), ord('B')]:
                        if key == ord('B'):
                            mode = 'baseline' if mode == 'raw' else 'raw'
                        mujoco.mj_resetData(model, data)
                        mujoco.mj_forward(model, data)
                        circuit.reset()
                        episode += 1
                        distance, collision_events, previous_collision = 0.0, 0, False
                        previous_position = data.xpos[robot_id, :2].copy()
                        paused = True
                        print(f'새 에피소드 {episode}, 모드 {mode}. Space로 시작.', flush=True)
                    elif key == ord('C'):
                        first_person = not first_person
                        with viewer.lock():
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED if first_person else mujoco.mjtCamera.mjCAMERA_TRACKING
                            viewer.cam.fixedcamid = 0
                            viewer.cam.trackbodyid = robot_id
                if not paused:
                    renderer.update_scene(data, camera='eye')
                    rgb = renderer.render().copy()
                    scores = circuit.step(rgb, args.neural_steps)
                    speed, turn = raw_command(scores, args.cruise, args.sensitivity) if mode == 'raw' else visual_baseline(rgb)
                    motors = np.clip([(speed - turn * 0.13) / 0.09, (speed + turn * 0.13) / 0.09], -15, 15)
                    data.ctrl[:] = motors
                    contact = False
                    for _ in range(10):
                        mujoco.mj_step(model, data)
                        for c in data.contact[:data.ncon]:
                            if (int(c.geom1) in obstacles) != (int(c.geom2) in obstacles):
                                contact = True
                    if contact and not previous_collision:
                        collision_events += 1
                    previous_collision = contact
                    position = data.xpos[robot_id, :2].copy()
                    distance += float(np.linalg.norm(position - previous_position))
                    previous_position = position
                    ms = (time.perf_counter() - start) * 1000
                    writer.writerow(dict(zip(columns, [episode, mode, data.time, *position, distance, int(contact), collision_events,
                                                        scores['L'], scores['R'], speed, turn, *motors, ms])))
                    if time.perf_counter() - last_print > 1:
                        print(f'{mode} | {data.time:.1f}s | DN L={scores["L"]:.4f} R={scores["R"]:.4f} | '
                              f'이동 {distance:.2f}m | 충돌 {collision_events} | 계산 {ms:.0f}ms', flush=True)
                        log.flush()
                        last_print = time.perf_counter()
                # GUI label shows experiment state without modifying control inputs.
                with viewer.lock():
                    viewer.user_scn.ngeom = 1
                    geom = viewer.user_scn.geoms[0]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([0.001, 0.001, 0.001]),
                                       data.xpos[robot_id] + np.array([0., 0., 0.35]),
                                       np.eye(3).ravel(), np.array([1., 1., 1., 1.], dtype=np.float32))
                    geom.label = f'{mode.upper()} {"PAUSED" if paused else "RUN"} hits={collision_events}'
                viewer.sync()
                time.sleep(max(0, 0.05 - (time.perf_counter() - start)))
    finally:
        renderer.close()
        print(f'종료. 실험 기록: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuit', type=Path, default=ROOT / 'navigation_circuit')
    parser.add_argument('--mode', choices=['raw', 'baseline'], default='raw')
    parser.add_argument('--nt', choices=['structural', 'ach-gaba'], default='structural')
    parser.add_argument('--cruise', type=float, default=0.18)
    parser.add_argument('--sensitivity', type=float, default=4.0)
    parser.add_argument('--neural-steps', type=int, default=4)
    args = parser.parse_args()
    if args.neural_steps < 1 or args.cruise < 0 or args.sensitivity < 0:
        parser.error('neural-steps must be positive; cruise and sensitivity must be nonnegative')
    run(args)
