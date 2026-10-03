"""Whole CNS drives an untethered public fly body; records before circuit selection."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
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
    raise SystemExit('install_navigation.cmd로 MuJoCo를 설치하세요.') from exc
from full_cns_model import FullCNS
from full_cns_recording import Recorder
from fly_motor_decoder import FlyMotorDecoder
from full_fly_gui import FlyPanel
from prepare_fly_body import DEST, LEGS

CASES = ('neutral', 'bright', 'dim', 'move_left', 'move_right', 'approach', 'free')
BAD_WARNINGS = ('mjWARN_INERTIA', 'mjWARN_BADQPOS', 'mjWARN_BADQVEL', 'mjWARN_BADQACC', 'mjWARN_BADCTRL')


def bad_values(values):
    """Same magnitude boundary as MuJoCo, including finite but exploded values."""
    return bool(np.any(~np.isfinite(values)) or np.any(np.abs(values) > mujoco.mjMAXVAL))


def named(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f'초파리 몸체 모델에 이름이 없습니다: {name}')
    return index


def environment(model, data, case, seconds, onset):
    brightness, target = 1., [8., 0., -20.]
    age = max(0., seconds-onset)
    if case == 'bright' and seconds >= onset:
        brightness = 1.8
    elif case == 'dim' and seconds >= onset:
        brightness = .15
    elif case in ('move_left', 'move_right'):
        y = -3. + 6.*((age % 2.)/2.)
        if case == 'move_right':
            y = -y
        target = [8., y, 1.5] if seconds >= onset else [8., 0., -20.]
    elif case == 'approach' and seconds >= onset:
        target = [max(1.5, 10.-4.*age), 0., 1.5]
    elif case == 'free' and seconds >= onset:
        target = [8., 3.*np.sin(age*3.), 1.5]
    data.mocap_pos[0] = target
    model.light_diffuse[:] = .7 * brightness
    model.vis.headlight.diffuse[:] = .4 * brightness
    model.vis.headlight.ambient[:] = .3 * brightness
    return {'target_xyz_mm': target, 'light_multiplier': brightness, 'onset_s': onset,
            'active': bool(seconds >= onset), 'coordinate_frame': 'fixed world, not body-relative'}


def reset_body(model, data, body_manifest, root, free_joint, foot_geoms):
    mujoco.mj_resetData(model, data)
    for name, value in body_manifest['initial_joint_angles_rad'].items():
        joint = named(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    for name, quaternion in body_manifest.get('initial_ball_quaternions_wxyz', {}).items():
        joint = named(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        address = int(model.jnt_qposadr[joint])
        data.qpos[address:address+4] = quaternion
    address = int(model.jnt_qposadr[free_joint])
    mujoco.mj_forward(model, data)
    # Place the published static pose just above the floor. This is initialization only.
    minimum_z = float('inf')
    for geom in foot_geoms:
        mesh = int(model.geom_dataid[geom])
        if mesh >= 0:
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start:start+count]
            transformed = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            minimum_z = min(minimum_z, float(transformed[:, 2].min()))
    data.qpos[address:address+2] -= data.xpos[root, :2]
    if np.isfinite(minimum_z):
        data.qpos[address+2] += .01-minimum_z
    data.qvel[:] = 0.
    data.ctrl[:] = 0.
    mujoco.mj_forward(model, data)


def contacts(model, data, floor):
    items = []
    ground = {leg: False for leg in LEGS}
    for index in range(data.ncon):
        contact = data.contact[index]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        frame = contact.frame.reshape(3, 3)
        item = {'geom_ids': [int(contact.geom1), int(contact.geom2)],
                'position_mm': contact.pos.tolist(), 'distance_mm': float(contact.dist),
                'frame_world_rows': frame.tolist(), 'force_contact_uN': force[:3].tolist(),
                'torque_contact_nNm': force[3:].tolist(),
                'force_world_uN': (frame.T @ force[:3]).tolist()}
        items.append(item)
        if floor in item['geom_ids'] and force[0] > 0:
            other = contact.geom2 if contact.geom1 == floor else contact.geom1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ''
            for leg in LEGS:
                if name.startswith(leg + 'Tarsus'):
                    ground[leg] = True
    return items, ground


def body_observation(model, data, root, previous_xy, path_mm):
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root, velocity, 0)
    axis = data.xmat[root].reshape(3, 3)[:, 0]
    xyz = data.xpos[root].copy()
    if previous_xy is not None:
        path_mm += float(np.linalg.norm(xyz[:2]-previous_xy))
    return {'xyz_mm': xyz.tolist(), 'quaternion_wxyz': data.xquat[root].tolist(),
            'linear_velocity_world_mm_s': velocity[3:].tolist(),
            'angular_velocity_world_rad_s': velocity[:3].tolist(),
            'forward_axis_world': axis.tolist(), 'forward_velocity_mm_s': float(axis @ velocity[3:]),
            'up_axis_world': data.xmat[root].reshape(3, 3)[:, 2].tolist(), 'path_mm': path_mm}


class PhysicsTrace:
    """Every integration step: body, joints, actuator/passive/contact forces and contacts."""
    def __init__(self):
        self.samples, self.contact_rows, self.offsets = [], [], [0]

    def append(self, model, data, outputs, contact_items):
        self.samples.append({
            'time_s': float(data.time), 'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(),
            'qacc': data.qacc.copy(), 'ctrl': data.ctrl.copy(), 'actuator_force_nNm': data.actuator_force.copy(),
            'qfrc_actuator_native': data.qfrc_actuator.copy(), 'qfrc_passive_native': data.qfrc_passive.copy(),
            'qfrc_constraint': data.qfrc_constraint.copy(), 'body_xpos_mm': data.xpos.copy(),
            'body_xquat': data.xquat.copy(), 'activation': np.asarray(outputs.activation).copy(),
            'channel_force_uN': np.asarray(outputs.forces).copy(),
            'warning_counts': np.asarray([w.number for w in data.warning], dtype=np.int64),
            'mocap_pos_mm': data.mocap_pos.copy(), 'mocap_quat': data.mocap_quat.copy(),
            'light_diffuse': model.light_diffuse.copy(),
            'headlight_diffuse': model.vis.headlight.diffuse.copy(),
            'headlight_ambient': model.vis.headlight.ambient.copy(),
        })
        for item in contact_items:
            self.contact_rows.append(item)
        self.offsets.append(len(self.contact_rows))

    def arrays(self):
        result = {f'physics_{key}': np.asarray([row[key] for row in self.samples]) for key in self.samples[0]}
        result['physics_contact_offsets'] = np.asarray(self.offsets, dtype=np.int64)
        result['physics_contact_geom_ids'] = np.asarray([c['geom_ids'] for c in self.contact_rows], dtype=np.int32).reshape(-1, 2)
        result['physics_contact_position_mm'] = np.asarray([c['position_mm'] for c in self.contact_rows]).reshape(-1, 3)
        result['physics_contact_distance_mm'] = np.asarray([c['distance_mm'] for c in self.contact_rows])
        result['physics_contact_frame'] = np.asarray([c['frame_world_rows'] for c in self.contact_rows]).reshape(-1, 3, 3)
        result['physics_contact_force_uN'] = np.asarray([c['force_contact_uN'] for c in self.contact_rows]).reshape(-1, 3)
        result['physics_contact_torque_nNm'] = np.asarray([c['torque_contact_nNm'] for c in self.contact_rows]).reshape(-1, 3)
        result['physics_contact_force_world_uN'] = np.asarray([c['force_world_uN'] for c in self.contact_rows]).reshape(-1, 3)
        return result


def physics_failure(output, model, data, circuit, decoder, rgb, trace, before, episode, case):
    """Preserve the failed substep separately; never label it a completed valid frame."""
    if (output / 'failure.json').exists():
        archive = output / 'failures' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        archive.mkdir(parents=True)
        for filename in ('failure.json', 'failure_body.npz', 'failure_neural_state.npy'):
            if (output / filename).exists():
                (output / filename).rename(archive / filename)
    warnings = []
    for name in BAD_WARNINGS:
        slot = int(getattr(mujoco.mjtWarning, name))
        warning = data.warning[slot]
        if warning.number:
            index, joint_name = int(warning.lastinfo), None
            if name in ('mjWARN_BADQVEL', 'mjWARN_BADQACC') and 0 <= index < model.nv:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.dof_jntid[index]))
            elif name == 'mjWARN_BADQPOS' and 0 <= index < model.nq:
                joint = int(np.searchsorted(model.jnt_qposadr, index, side='right')-1)
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            warnings.append({'type': name, 'count': int(warning.number), 'index': index, 'joint': joint_name})
    if not warnings:
        for label, values in (('qacc', data.qacc), ('qvel', data.qvel), ('qpos', data.qpos)):
            invalid = np.flatnonzero(~np.isfinite(values) | (np.abs(values) > mujoco.mjMAXVAL))
            if len(invalid):
                index = int(invalid[0])
                joint = (int(model.dof_jntid[index]) if label != 'qpos' else
                         int(np.searchsorted(model.jnt_qposadr, index, side='right')-1))
                warnings.append({'type': 'invalid_' + label, 'source': 'local MuJoCo magnitude-bound check',
                                 'index': index, 'joint': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)})
    arrays = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), qacc=data.qacc.copy(), ctrl=data.ctrl.copy(),
                  qfrc_passive=data.qfrc_passive.copy(), qfrc_actuator=data.qfrc_actuator.copy(),
                  rgb=rgb, channel_activity=decoder.raw.copy(), channel_force_uN=decoder.forces.copy(),
                  joint_torque_nNm=decoder.torques.copy(),
                  before_qpos=before['qpos'], before_qvel=before['qvel'])
    if trace.samples:
        arrays.update(trace.arrays())
    np.savez(output / 'failure_body.npz', **arrays)
    np.save(output / 'failure_neural_state.npy', circuit.state)
    details = {'episode': episode, 'case': case, 'neural_tick': circuit.tick,
               'before_sim_seconds': before['time'], 'after_sim_seconds': float(data.time),
               'completed_physics_steps_in_failed_frame': len(trace.samples), 'warnings': warnings,
               'neural_state': 'failure_neural_state.npy; dense lossless state in original graph index order',
               'body_state': 'failure_body.npz; partial interval, not a valid completed frame',
               'max_abs_control_nNm': float(np.max(np.abs(data.ctrl))),
               'note': 'Failure preserved before raising. Reproduction/simulation was not performed by the analysis tool.'}
    (output / 'failure.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    return warnings


def run(args):
    if not (DEST / 'manifest.json').exists():
        raise FileNotFoundError('먼저 prepare_fly_body.py로 공개 초파리 몸체를 준비하세요.')
    body_manifest = json.loads((DEST / 'manifest.json').read_text(encoding='utf-8'))
    arena_bytes = (DEST / 'arena.xml').read_bytes()
    if hashlib.sha256(arena_bytes).hexdigest() != body_manifest['derived_xml_sha256']:
        raise ValueError('몸체 XML이 manifest와 다릅니다. prepare_fly_body.py --offline으로 다시 준비하세요.')
    print('공개 초파리 몸체와 전체 CNS를 구성합니다. 전체 segment 계산에는 시간이 걸립니다.', flush=True)
    circuit = FullCNS(args.circuit, glutamate=args.glutamate, gain=args.gain,
                      leak=args.leak, weight_mode=args.weight_mode)
    decoder = FlyMotorDecoder(circuit.frame, args.mapping, args.activity_scale, args.muscle_force_uN, args.muscle_tau)
    model = mujoco.MjModel.from_xml_path(str(DEST / 'arena.xml'))
    physics = body_manifest.get('physics_assumptions')
    if physics is None:
        raise ValueError('몸체 설정이 이전 버전입니다. run_mujoco.cmd로 새 XML을 준비하세요.')
    for name in body_manifest['stabilized_foot_joint_names']:
        joint = named(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = int(model.jnt_dofadr[joint])
        model.dof_damping[dof] = max(float(model.dof_damping[dof]),
                                    physics['distal_foot']['minimum_damping_nNm_s_per_rad'])
    # Stop on the first bad state and retain diagnostics rather than losing them to auto reset.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    data = mujoco.MjData(model)
    root = named(model, mujoco.mjtObj.mjOBJ_BODY, 'Thorax')
    free_joint = named(model, mujoco.mjtObj.mjOBJ_JOINT, 'fly_free')
    floor = named(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    foot_bodies = {leg: named(model, mujoco.mjtObj.mjOBJ_BODY, leg+'Tarsus5') for leg in LEGS}
    foot_geoms = [named(model, mujoco.mjtObj.mjOBJ_GEOM, leg+'Tarsus'+str(i)) for leg in LEGS for i in range(1, 6)]
    joints = np.array([named(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in decoder.joint_names])
    actuators = np.array([named(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n+'_CNS') for n in decoder.joint_names])
    qindices, vindices = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    substeps = int(round(args.neural_interval / model.opt.timestep))
    if substeps < 1 or not np.isclose(substeps * model.opt.timestep, args.neural_interval, rtol=0, atol=1e-9):
        raise ValueError('neural-interval은 몸체 timestep(0.0001초)의 정수 배수여야 합니다.')
    reset_body(model, data, body_manifest, root, free_joint, foot_geoms)
    output = ROOT / 'simulation_logs' / ('full_fly_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    recorder = Recorder(output, args.record_gb)
    commands, command_lock = [], threading.Lock()
    def enqueue(key):
        with command_lock:
            commands.append(key)
    renderer = panel = None
    case, episode, paused, first_person = args.case, 0, True, False
    fault_message = ''
    neutral_frames = {}
    path_mm, previous_xy = 0., data.xpos[root, :2].copy()
    def observation():
        items, grounded = contacts(model, data, floor)
        feet = {leg: {'xyz_mm': data.xpos[foot_bodies[leg]].tolist(), 'ground_contact': grounded[leg]} for leg in LEGS}
        body = body_observation(model, data, root, previous_xy, path_mm)
        decoded = decoder.snapshot(data.qpos[qindices], data.qvel[vindices])
        return decoded, body, feet, items
    try:
        # Freeze source/mapping/calibration and compiled body for interpretation of this run.
        assets = output / 'body_assets'
        shutil.copytree(DEST / 'upstream', assets / 'upstream')
        shutil.copy2(DEST / 'arena.xml', assets / 'arena.xml')
        shutil.copy2(DEST / 'manifest.json', assets / 'manifest.json')
        shutil.copy2(args.mapping, output / 'motor_mapping.json')
        source_folder = output / 'source'
        source_folder.mkdir()
        for filename in ('mujoco_full_fly.py', 'full_cns_model.py', 'fly_motor_decoder.py',
                         'full_cns_recording.py', 'prepare_fly_body.py', 'full_fly_gui.py'):
            shutil.copy2(ROOT / 'src' / filename, source_folder / filename)
        mujoco.mj_saveModel(model, str(output / 'body_model.mjb'), None)
        np.save(output / 'initial_qpos.npy', data.qpos)
        circuit.frame.to_feather(output / 'annotations.feather')
        decoder.all_motor.to_csv(output / 'all_leg_motor_neurons.csv', index=False, encoding='utf-8-sig')
        decoder.unmapped.to_csv(output / 'unmapped_leg_motor_neurons.csv', index=False, encoding='utf-8-sig')
        metadata = {
            'schema': 2, 'output_mode': 'public_fly_body', 'circuit_folder': str(args.circuit.resolve()),
            'circuit_report': json.loads((args.circuit / 'report.json').read_text(encoding='utf-8')),
            'nt': circuit.nt_report, 'inputs': circuit.input_report, 'motor_population_counts': circuit.motor_report,
            'parameters': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            'equation': 'h_next=leak*h+(1-leak)*max(0,tanh(gain*W*h+camera_input))',
            'body_source': body_manifest, 'muscle_decoder': decoder.report,
            'mujoco_version': mujoco.__version__, 'neural_tick_seconds': args.neural_interval,
            'neural_time_calibration': 'engineering interval; not measured biological timing',
            'physics_step_seconds': float(model.opt.timestep), 'physics_steps_per_neural_tick': substeps,
            'recording': 'Lossless whole graph every neural tick; camera before update; every physics step in same snapshot.',
            'model_names': {label: [mujoco.mj_id2name(model, kind, i) for i in range(count)] for label, kind, count in
                            [('bodies', mujoco.mjtObj.mjOBJ_BODY, model.nbody), ('joints', mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
                             ('geoms', mujoco.mjtObj.mjOBJ_GEOM, model.ngeom), ('actuators', mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)]},
            'joint_qpos_addresses': model.jnt_qposadr.tolist(), 'joint_dof_addresses': model.jnt_dofadr.tolist(),
            'joint_axes_local': model.jnt_axis.tolist(),
            'compiled_joint_stiffness_nNm_per_rad': model.jnt_stiffness.tolist(),
            'compiled_dof_damping_native': model.dof_damping.tolist(),
            'numerical_failure_policy': 'Auto reset disabled; archive failure and pause with last completed body frame. Reset required to resume.',
            'body_root_id': root, 'floor_geom_id': floor, 'foot_body_ids': foot_bodies,
            'generalized_force_units': 'Free-root translation uN; free-root rotation and all hinge DoFs nNm. Direct hinge actuator_force nNm.',
            'sensory_input': 'One camera to existing L1/L2/L3 engineering adapter. Other sensory neurons retained, not externally driven.',
            'body_feedback': 'Recorded only; no inferred proprioceptive injection.',
            'fixed_cruise': False, 'baseline_controller': False, 'learning': False, 'gait_controller': False,
            'neutral_comparison': 'Diagnostics at equal ticks after complete resets; no effect on output.',
            'acquisition_manifest': json.loads((ROOT / 'data/male-cns-v1.0/manifest.json').read_text(encoding='utf-8')),
            'interpretation': 'Body physics and motor adapter are hypotheses. Observed movement is not proof of intact fly behavior.',
        }
        (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'기록: {output}', flush=True)
        print(f'몸체 회전 자유도 {body_manifest.get("rotational_degrees_of_freedom", model.njnt-1)}개 + 자유 몸체, '
              f'관절 {model.njnt-1}개, CNS 토크 출력 장치 {len(joints)}개', flush=True)
        print(f'다리 MN 출력 대응 {decoder.report["mapped_mn_count"]}/{decoder.report["leg_mn_count"]}; 미대응도 모두 저장합니다.', flush=True)
        print('Space 시작/정지, R 초기화, C 시점, 1 물체 없음, 2 밝게, 3 어둡게, 4 왼쪽, 5 오른쪽, 6 접근, 7 이동 물체', flush=True)
        print(f'자극은 가상 {args.stimulus_onset:g}초에 시작합니다. 전체 상태 매 업데이트 저장, 한도 {args.record_gb:g} GiB.', flush=True)
        renderer = mujoco.Renderer(model, height=120, width=160)
        panel = FlyPanel(enqueue, decoder)
        decoded, body, feet, _ = observation()
        panel.refresh(decoded, body, feet)
        with mujoco.viewer.launch_passive(model, data, key_callback=enqueue) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = root
                viewer.cam.distance = 7.
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -30
            while viewer.is_running() and not panel.closed:
                start = time.perf_counter()
                panel.pump()
                if panel.closed:
                    break
                with command_lock:
                    keys = commands[:]
                    commands.clear()
                for key in keys:
                    if key == 32:
                        if not fault_message:
                            paused = not paused
                    elif key == ord('R') or ord('1') <= key <= ord('7'):
                        if key != ord('R'):
                            case = CASES[key-ord('1')]
                        reset_body(model, data, body_manifest, root, free_joint, foot_geoms)
                        fault_message = ''
                        circuit.reset()
                        decoder.reset()
                        episode += 1
                        paused, path_mm, previous_xy = True, 0., data.xpos[root, :2].copy()
                        if case == 'neutral':
                            neutral_frames.clear()
                        decoded, body, feet, _ = observation()
                        panel.refresh(decoded, body, feet)
                    elif key == ord('C'):
                        first_person = not first_person
                        with viewer.lock():
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED if first_person else mujoco.mjtCamera.mjCAMERA_TRACKING
                            viewer.cam.fixedcamid = named(model, mujoco.mjtObj.mjOBJ_CAMERA, 'eye')
                            viewer.cam.trackbodyid = root
                if fault_message:
                    panel.set_status(case, True, recorder.count, data.time, fault_message)
                    viewer.sync()
                    time.sleep(.02)
                    continue
                stimulus = environment(model, data, case, float(data.time), args.stimulus_onset)
                mujoco.mj_forward(model, data)
                panel.set_status(case, paused, recorder.count, data.time)
                viewer.sync()
                if paused:
                    time.sleep(.02)
                    continue
                panel.set_status(case, False, recorder.count, data.time, '전체 CNS 계산 중')
                panel.pump()
                input_time = float(data.time)
                renderer.update_scene(data, camera='eye')
                rgb = renderer.render().copy()
                input_qpos = data.qpos.copy()
                input_body = body_observation(model, data, root, None, path_mm)
                neural_started = time.perf_counter()
                scores = circuit.step(rgb)
                neural_ms = (time.perf_counter()-neural_started)*1000
                decoder.observe(circuit.state)
                trace = PhysicsTrace()
                # Capture complete integration state, including warmstart, for display rollback.
                state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
                display_state = np.empty(mujoco.mj_stateSize(model, state_spec))
                mujoco.mj_getState(model, data, display_state, state_spec)
                input_path, input_xy = path_mm, previous_xy.copy()
                physics_started = time.perf_counter()
                for _ in range(substeps):
                    environment(model, data, case, float(data.time), args.stimulus_onset)
                    data.ctrl[actuators] = decoder.advance(float(model.opt.timestep))
                    before = {'time': float(data.time), 'qpos': data.qpos.copy(), 'qvel': data.qvel.copy()}
                    counts_before = {name: int(data.warning[int(getattr(mujoco.mjtWarning, name))].number) for name in BAD_WARNINGS}
                    mujoco.mj_step(model, data)
                    bad = [name for name in BAD_WARNINGS if data.warning[int(getattr(mujoco.mjtWarning, name))].number > counts_before[name]]
                    if bad or data.time <= before['time'] or bad_values(data.qpos) or bad_values(data.qvel):
                        warnings = physics_failure(output, model, data, circuit, decoder, rgb, trace, before, episode, case)
                        fault_message = f'가상 {data.time:.4f}s 수치 오류: {warnings} — R 또는 조건 선택으로 초기화'
                        break
                    mujoco.mj_forward(model, data)
                    bad = [name for name in BAD_WARNINGS if data.warning[int(getattr(mujoco.mjtWarning, name))].number > counts_before[name]]
                    if bad or bad_values(data.qacc):
                        warnings = physics_failure(output, model, data, circuit, decoder, rgb, trace, before, episode, case)
                        fault_message = f'가상 {data.time:.4f}s 수치 오류: {warnings} — R 또는 조건 선택으로 초기화'
                        break
                    contact_items, _ = contacts(model, data, floor)
                    trace.append(model, data, decoder, contact_items)
                    path_mm += float(np.linalg.norm(data.xpos[root, :2]-previous_xy))
                    previous_xy = data.xpos[root, :2].copy()
                    if len(trace.samples) % 50 == 0:
                        panel.pump()
                        if panel.closed or not viewer.is_running():
                            break
                        viewer.sync()
                physics_ms = (time.perf_counter()-physics_started)*1000
                if panel.closed or not viewer.is_running():
                    break
                if fault_message:
                    error_log = ROOT / 'logs' / 'full_cns_error.log'
                    error_log.parent.mkdir(parents=True, exist_ok=True)
                    error_log.write_text(f'{fault_message}\n상세 기록: {output / "failure.json"}\n', encoding='utf-8')
                    print(f'{fault_message}; 기록: {output / "failure.json"}', flush=True)
                    mujoco.mj_setState(model, data, display_state, state_spec)
                    environment(model, data, case, input_time, args.stimulus_onset)
                    mujoco.mj_forward(model, data)
                    path_mm, previous_xy, paused = input_path, input_xy, True
                    panel.set_status(case, True, recorder.count, data.time, fault_message)
                    continue
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError('몸체 수치 상태가 유한하지 않습니다. 활동 기준/힘 설정을 확인하세요.')
                decoded, body, feet, contact_items = observation()
                if case == 'neutral':
                    neutral_frames[circuit.tick] = {'episode': episode, 'outputs': decoded, 'body': body}
                reference = neutral_frames.get(circuit.tick) if case != 'neutral' else None
                event = {
                    'episode': episode, 'case': case, 'output_mode': 'public_fly_body',
                    'sim_seconds': float(data.time), 'input_sim_seconds': input_time, 'neural_tick': circuit.tick,
                    'stimulus': stimulus, 'body': body, 'input_body': input_body, 'feet': feet, 'contacts': contact_items,
                    'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(), 'motor_outputs': decoded,
                    'motor_scores': scores, 'motor_scores_role': 'diagnostic only',
                    'motor_commands': decoder.torques.tolist(), 'motor_command_unit': 'nNm',
                    'robot_xy': data.xpos[root, :2].tolist(), 'robot_xy_unit': 'mm', 'path_m': path_mm * .001,
                    'neutral_reference_episode': reference['episode'] if reference else None,
                    'neutral_angle_difference_rad': (np.asarray(decoded['angle_rad'])-np.asarray(reference['outputs']['angle_rad'])).tolist() if reference else None,
                    'neural_compute_ms': neural_ms,
                    'physics_trace_ms': physics_ms,
                    'compute_ms_excluding_record': (time.perf_counter()-start)*1000,
                }
                arrays = trace.arrays()
                arrays.update(input_qpos=input_qpos, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                              channel_names=np.asarray(decoder.names), joint_names=np.asarray(decoder.joint_names),
                              channel_activity=decoder.raw.copy(), channel_target_activation=decoder.targets.copy(),
                              channel_activation=decoder.activation.copy(), channel_force_uN=decoder.forces.copy(),
                              joint_torque_nNm=decoder.torques.copy(), joint_angle_rad=data.qpos[qindices].copy())
                panel.refresh(decoded, body, feet, reference['outputs'] if reference else None, rgb)
                panel.set_status(case, False, recorder.count, data.time, '전체 신경 · 몸체 상태 저장 중')
                panel.pump()
                viewer.sync()
                exhausted = recorder.save(circuit, rgb, event, extra_arrays=arrays)
                print(f'{case} | 가상 {data.time:.3f}s | 앞 방향 속도 {body["forward_velocity_mm_s"]:+.4f} mm/s | '
                      f'최대 토크 {np.max(np.abs(decoder.torques)):.4f} nNm | CNS {neural_ms:.0f} ms | '
                      f'저장 {recorder.bytes/1024**3:.2f} GiB', flush=True)
                if exhausted:
                    print('전체 상태 저장 한도에 도달했습니다. --record-gb로 한도를 변경할 수 있습니다.', flush=True)
                    break
    finally:
        if renderer is not None:
            renderer.close()
        if panel is not None:
            panel.close()
        recorder.close()
        print(f'기록 종료: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuit', type=Path, default=ROOT / 'full_cns')
    parser.add_argument('--mapping', type=Path, default=ROOT / 'config' / 'fly_motor_mapping.json')
    parser.add_argument('--case', choices=CASES, default='free')
    parser.add_argument('--glutamate', choices=('positive', 'negative'), default='positive')
    parser.add_argument('--gain', type=float, default=.95)
    parser.add_argument('--leak', type=float, default=.65)
    parser.add_argument('--weight-mode', choices=('incoming', 'raw'), default='incoming')
    parser.add_argument('--activity-scale', type=float, default=1e-4)
    parser.add_argument('--muscle-force-uN', dest='muscle_force_uN', type=float, default=40.)
    parser.add_argument('--muscle-tau', type=float, default=.1)
    parser.add_argument('--neural-interval', type=float, default=.05)
    parser.add_argument('--stimulus-onset', type=float, default=.5)
    parser.add_argument('--record-gb', type=float, default=10.)
    args = parser.parse_args()
    numbers = [args.gain, args.leak, args.activity_scale, args.muscle_force_uN, args.muscle_tau,
               args.neural_interval, args.stimulus_onset, args.record_gb]
    if not all(np.isfinite(x) for x in numbers) or not 0 <= args.leak < 1 or args.gain < 0 or args.stimulus_onset < 0 or any(x <= 0 for x in [args.activity_scale, args.muscle_force_uN, args.muscle_tau, args.neural_interval, args.record_gb]):
        parser.error('유한한 값이 필요합니다. leak은 [0,1), gain/자극 시작은 0 이상, 나머지는 양수여야 합니다.')
    try:
        run(args)
    except Exception:
        details = traceback.format_exc()
        error_log = ROOT / 'logs' / 'full_cns_error.log'
        error_log.parent.mkdir(parents=True, exist_ok=True)
        error_log.write_text(details, encoding='utf-8')
        print(details, file=sys.stderr, flush=True)
        print(f'오류 기록: {ROOT / "logs" / "full_cns_error.log"}', file=sys.stderr, flush=True)
        raise SystemExit(1)
