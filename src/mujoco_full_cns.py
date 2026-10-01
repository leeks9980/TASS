"""Whole-model MuJoCo observation run; recording precedes any circuit selection."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit('먼저 install_navigation.cmd로 MuJoCo를 설치하세요.') from exc
from full_cns_model import FullCNS
from full_cns_recording import Recorder
from full_cns_muscles import MuscleDecoder, LEGS, compare_to_neutral
from full_cns_muscle_gui import MusclePanel

CASES = ['neutral', 'bright', 'dim', 'move_left', 'move_right', 'approach', 'free']


def set_environment(model, data, case, seconds):
    brightness = 1.
    target = [0., 0., -10.]
    if case == 'bright':
        brightness = 1.8 if seconds >= 2 else 1.
    elif case == 'dim':
        brightness = 0.15 if seconds >= 2 else 1.
    elif case in ['move_left', 'move_right']:
        phase = (seconds % 8) / 8
        y = -0.9 + 1.8*phase
        if case == 'move_right':
            y = -y
        target = [1.5, y, 0.55]
    elif case == 'approach':
        target = [max(0.35, 2.2-0.15*seconds), 0., 0.55]
    elif case == 'free':
        target = [1.6, 0.7*np.sin(seconds*0.6), 0.55]
    data.mocap_pos[0] = target
    model.light_diffuse[:] = 0.7*brightness
    model.vis.headlight.diffuse[:] = 0.4*brightness
    model.vis.headlight.ambient[:] = 0.3*brightness
    mujoco.mj_forward(model, data)
    return {'target_xyz': target, 'light_multiplier': brightness}


def run(args):
    print('전체 CNS 구성 중: 미주석 segment를 포함합니다. 준비와 계산이 느릴 수 있습니다.', flush=True)
    circuit = FullCNS(args.circuit, glutamate=args.glutamate, gain=args.gain,
                      leak=args.leak, weight_mode=args.weight_mode)
    decoder = MuscleDecoder(circuit.frame, activity_scale=args.activity_scale,
                            force_max=args.muscle_force, moment_arm=args.moment_arm,
                            time_constant=args.muscle_tau)
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'models' / 'full_cns_muscle_arena.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, leg+'_joint') for leg in LEGS])
    actuator_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, leg+'_drive') for leg in LEGS])
    tibia_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, leg+'_tibia') for leg in LEGS])
    if np.any(joint_ids < 0) or np.any(actuator_ids < 0) or np.any(tibia_ids < 0) or robot < 0:
        raise ValueError('가상 근육 관절 모델의 이름과 출력 순서가 일치하지 않습니다.')
    qpos_indices, qvel_indices = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    output = ROOT / 'simulation_logs' / ('full_cns_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    recorder = Recorder(output, args.record_gb)
    metadata = {
        'circuit_folder': str(args.circuit.resolve()),
        'circuit_report': json.loads((args.circuit / 'report.json').read_text(encoding='utf-8')),
        'nt': circuit.nt_report, 'inputs': circuit.input_report, 'motor_population_counts': circuit.motor_report,
        'parameters': vars(args) | {'circuit': str(args.circuit)},
        'equation': 'h_next=leak*h+(1-leak)*max(0,tanh(gain*W*h+camera_input)); W uses recorded weight-mode and NT assumptions',
        'acquisition_manifest': json.loads((ROOT / 'data' / 'male-cns-v1.0' / 'manifest.json').read_text(encoding='utf-8')),
        'neural_tick_seconds': 'not biologically calibrated; one update per 0.05 simulated seconds',
        'recording': 'Every neural update; lossless uncompressed sparse/dense auto format; sparse omitted indices equal zero exactly',
        'visual_boundary': 'L1/L2/L3 engineering camera input; photoreceptors retained in graph but not directly driven',
        'output_mode': 'antagonistic_tibia_muscles',
        'body_output': 'annotated tibia flexor/extensor MN groups -> activation -> antagonist forces -> joint torques',
        'muscle_decoder': decoder.report,
        'rig_joint_names': [leg+'_joint' for leg in LEGS],
        'rig_actuator_names': [leg+'_drive' for leg in LEGS],
        'neutral_comparison': 'same neural tick after reset; diagnostics only, never fed to controller',
        'arena': 'full_cns_muscle_arena.xml',
        'fixed_cruise': False, 'baseline_controller': False, 'learning': False,
        'sensory_input': 'camera only; other sensory neurons retained but receive no external drive',
        'assets': json.loads((ROOT / 'data' / 'public_data_inventory.json').read_text(encoding='utf-8')),
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    circuit.frame.to_feather(output / 'annotations.feather')
    print(f'기록 폴더: {output}', flush=True)
    print(f'전체 graph ID {len(circuit.ids):,}, 주석 뉴런 {len(circuit.frame):,}, 연결 {circuit.matrix.nnz:,}', flush=True)
    print('Space 시작/정지, R 리셋, C 카메라, 1 무물체, 2 밝게, 3 어둡게, 4 왼쪽 이동, 5 오른쪽 이동, 6 접근, 7 자유 관찰', flush=True)
    print('전체 활동을 매 프레임 저장합니다. 저장 한도에 도달하면 종료됩니다.', flush=True)
    print('출력: 경골 굽힘/폄 가상 근육 → 여섯 관절 토크. 고정 몸체 관찰용이며 보행 모델은 아닙니다.', flush=True)
    print(f'활동 기준 {args.activity_scale:g}, 최대 근육 힘 {args.muscle_force:g} N, 모멘트암 {args.moment_arm:g} m', flush=True)
    if decoder.report['missing_groups']:
        print('출력 주석 누락(해당 근육 입력 0): '+', '.join(decoder.report['missing_groups']), flush=True)
    commands, command_lock = [], threading.Lock()
    def key_callback(key):
        with command_lock:
            commands.append(key)
    renderer = panel = None
    case, episode, paused, first_person = args.case, 0, True, False
    neutral_frames, neutral_episode = {}, None
    muscles = decoder.snapshot(data.qpos[qpos_indices], data.qvel[qvel_indices])
    def publish_view(viewer):
        with viewer.lock():
            viewer.user_scn.ngeom = 1+len(LEGS)
            for i in range(1+len(LEGS)):
                geom = viewer.user_scn.geoms[i]
                position = data.xpos[robot]+np.array([0., 0., .2]) if i == 0 else data.xpos[tibia_ids[i-1]]+np.array([0., 0., .08])
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.001]*3),
                                   position, np.eye(3).ravel(), np.array([1., 1., 1., 1.], dtype=np.float32))
                if i == 0:
                    geom.label = f'{case.upper()} {"PAUSED" if paused else "RUN"} frame={recorder.count}'
                else:
                    item = muscles[LEGS[i-1]]
                    geom.label = f'{LEGS[i-1]} q={np.degrees(item["angle_rad"]):+.1f} T={item["torque_Nm"]:+.3f}'
        viewer.sync()
    try:
        renderer = mujoco.Renderer(model, height=120, width=160)
        panel = MusclePanel(key_callback, decoder)
        panel.refresh(muscles, decoder.steering_values, None, None)
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = robot
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -35
            while viewer.is_running():
                start = time.perf_counter()
                panel.pump()
                with command_lock:
                    keys = commands[:]
                    commands.clear()
                for key in keys:
                    if key == 32:
                        paused = not paused
                        print('일시정지' if paused else '시작: 전체 신경망 계산과 기록을 진행합니다.', flush=True)
                    elif key == ord('R') or ord('1') <= key <= ord('7'):
                        if key != ord('R'):
                            case = CASES[key-ord('1')]
                        mujoco.mj_resetData(model, data)
                        circuit.reset()
                        decoder.reset()
                        mujoco.mj_forward(model, data)
                        episode += 1
                        if case == 'neutral':
                            neutral_frames.clear()
                            neutral_episode = None
                        muscles = decoder.snapshot(data.qpos[qpos_indices], data.qvel[qvel_indices])
                        panel.refresh(muscles, decoder.steering_values, None, neutral_episode)
                        paused = True
                        print(f'에피소드 {episode}: {case}, Space로 시작', flush=True)
                    elif key == ord('C'):
                        first_person = not first_person
                        with viewer.lock():
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED if first_person else mujoco.mjtCamera.mjCAMERA_TRACKING
                            viewer.cam.fixedcamid = 0
                            viewer.cam.trackbodyid = robot
                stimulus = set_environment(model, data, case, data.time)
                panel.set_status(case, paused, recorder.count, data.time)
                publish_view(viewer)
                if not paused:
                    print(f'프레임 {recorder.count}: 신경망 계산 중…', flush=True)
                    panel.set_status(case, paused, recorder.count, data.time, '전체 신경망 계산 중')
                    panel.pump()
                    renderer.update_scene(data, camera='eye')
                    rgb = renderer.render().copy()
                    scores = circuit.step(rgb)
                    decoder.observe(circuit.state)
                    for _ in range(10):
                        torques = decoder.advance(float(model.opt.timestep))
                        data.ctrl[actuator_ids] = torques
                        mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    muscles = decoder.snapshot(data.qpos[qpos_indices], data.qvel[qvel_indices])
                    if case == 'neutral':
                        neutral_frames[circuit.tick] = muscles
                        neutral_episode = episode
                    reference = compare_to_neutral(muscles, neutral_frames.get(circuit.tick))
                    event = {
                        'episode': episode, 'case': case, 'sim_seconds': float(data.time),
                        'input_sim_seconds': float(data.time-0.05), 'stimulus': stimulus,
                        'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(),
                        'robot_xy': data.xpos[robot, :2].tolist(), 'path_m': 0.,
                        'output_mode': 'antagonistic_tibia_muscles',
                        'motor_scores': scores, 'motor_scores_role': 'diagnostic only',
                        'motor_commands': torques.tolist(), 'motor_command_unit': 'Nm',
                        'neural_tick': circuit.tick, 'muscles': muscles,
                        'steering_diagnostics': decoder.steering_values,
                        'neutral_reference_episode': neutral_episode if reference is not None else None,
                        'neutral_difference': reference,
                        'compute_ms_excluding_record': (time.perf_counter()-start)*1000,
                    }
                    save_started = time.perf_counter()
                    print(f'프레임 {recorder.count}: 전체 상태 저장 중…', flush=True)
                    panel.refresh(muscles, decoder.steering_values, reference, neutral_episode, rgb)
                    panel.set_status(case, paused, recorder.count, data.time, '전체 상태 저장 중')
                    panel.pump()
                    publish_view(viewer)
                    exhausted = recorder.save(circuit, rgb, event)
                    save_ms = (time.perf_counter()-save_started)*1000
                    elapsed_ms = (time.perf_counter()-start)*1000
                    print(f'{case} | {data.time:.2f}s | MN L={scores["L"]:.3e} R={scores["R"]:.3e} | '
                          f'최대 토크 {np.max(np.abs(torques)):.4f}Nm | 계산 {event["compute_ms_excluding_record"]:.0f}ms | '
                          f'저장 {save_ms:.0f}ms | 전체 {elapsed_ms:.0f}ms | 기록 {recorder.bytes/1024**3:.2f}GB', flush=True)
                    if exhausted:
                        print('기록 저장 한도에 도달하여 종료합니다. --record-gb로 한도를 변경할 수 있습니다.', flush=True)
                        break
                panel.set_status(case, paused, recorder.count, data.time)
                publish_view(viewer)
                time.sleep(max(0., 0.05-(time.perf_counter()-start)))
    finally:
        if renderer is not None:
            renderer.close()
        if panel is not None:
            panel.close()
        recorder.close()
        print(f'관찰 종료. 기록: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuit', type=Path, default=ROOT / 'full_cns')
    parser.add_argument('--case', choices=CASES, default='free')
    parser.add_argument('--glutamate', choices=['positive', 'negative'], default='positive')
    parser.add_argument('--gain', type=float, default=0.95)
    parser.add_argument('--weight-mode', choices=['incoming', 'raw'], default='incoming')
    parser.add_argument('--leak', type=float, default=0.65)
    parser.add_argument('--activity-scale', type=float, default=1e-4, help='Fixed neural activity corresponding to full muscle target activation.')
    parser.add_argument('--muscle-force', type=float, default=4., help='Force at full activation, N (engineering assumption).')
    parser.add_argument('--moment-arm', type=float, default=.04, help='Antagonist moment arm, m.')
    parser.add_argument('--muscle-tau', type=float, default=.1, help='Activation time constant, simulated seconds.')
    parser.add_argument('--record-gb', type=float, default=10.)
    args = parser.parse_args()
    values = [args.leak, args.gain, args.record_gb, args.activity_scale, args.muscle_force, args.moment_arm, args.muscle_tau]
    if not all(np.isfinite(x) for x in values) or not 0 <= args.leak < 1 or args.gain < 0 or any(x <= 0 for x in values[2:]):
        parser.error('finite parameters required; leak must be [0,1), gain nonnegative, recording/muscle parameters positive')
    if args.muscle_force * args.moment_arm > 2:
        parser.error('muscle-force * moment-arm must not exceed the rig actuator limit of 2 Nm')
    if not (args.circuit / 'report.json').exists():
        parser.error('먼저 prepare_full_cns.py 또는 run_full_cns.cmd를 실행하세요.')
    try:
        run(args)
    except Exception:
        details = traceback.format_exc()
        error_log = ROOT / 'logs' / 'full_cns_error.log'
        error_log.parent.mkdir(parents=True, exist_ok=True)
        error_log.write_text(details, encoding='utf-8')
        print(details, file=sys.stderr, flush=True)
        print(f'오류 내용을 {ROOT / "logs" / "full_cns_error.log"}에 저장했습니다.', file=sys.stderr, flush=True)
        raise SystemExit(1)
