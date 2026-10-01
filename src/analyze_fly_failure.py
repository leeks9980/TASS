"""Read saved partial states to locate onset of a numerical failure. No stepping."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import mujoco


def analyze(run):
    model = mujoco.MjModel.from_binary_path(str(run / 'body_model.mjb'))
    failure = json.loads((run / 'failure.json').read_text(encoding='utf-8'))
    with np.load(run / 'failure_body.npz') as snapshot:
        times = snapshot['physics_time_s']
        acceleration = snapshot['physics_qacc']
        velocity = snapshot['physics_qvel']
        qpos = snapshot['physics_qpos']
        ctrl = snapshot['physics_ctrl']
        passive = snapshot['physics_qfrc_passive_native']
        contact_offsets = snapshot['physics_contact_offsets']
        contact_ids = snapshot['physics_contact_geom_ids']
        contact_forces = snapshot['physics_contact_force_uN']
        rows = []
        for dof in range(model.nv):
            joint = int(model.dof_jntid[dof])
            maximum = int(np.argmax(np.abs(acceleration[:, dof])))
            rows.append({'dof': dof, 'joint': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                         'max_abs_acceleration': float(np.max(np.abs(acceleration[:, dof]))),
                         'max_abs_velocity': float(np.max(np.abs(velocity[:, dof]))),
                         'time_max_acceleration_s': float(times[maximum]),
                         'max_abs_passive_native': float(np.max(np.abs(passive[:, dof]))),
                         'joint_stiffness': float(model.jnt_stiffness[joint]),
                         'dof_damping': float(model.dof_damping[dof])})
        rows.sort(key=lambda x: x['max_abs_acceleration'], reverse=True)
        timeline = []
        for time in np.linspace(float(times[0]), float(times[-1]), min(8, len(times))):
            i = int(np.argmin(np.abs(times-time)))
            order = np.argsort(np.abs(acceleration[i]))[-3:][::-1]
            a, b = int(contact_offsets[i]), int(contact_offsets[i+1])
            timeline.append({'time_s': float(times[i]), 'max_abs_qvel': float(np.max(np.abs(velocity[i]))),
                             'max_abs_qacc': float(np.max(np.abs(acceleration[i]))),
                             'worst_dofs': [{'dof': int(dof), 'joint': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.dof_jntid[dof])),
                                             'acceleration': float(acceleration[i, dof])} for dof in order],
                             'contact_pairs': [[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) for geom in pair] for pair in contact_ids[a:b]],
                             'max_contact_force_uN': float(np.max(np.abs(contact_forces[a:b]))) if b > a else 0.})
        positions = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
                     [float(qpos[0, model.jnt_qposadr[i]]), float(qpos[-1, model.jnt_qposadr[i]])]
                     for i in range(model.njnt) if int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_HINGE) and 'Pedicel' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or '')}
        result = {'failure': failure, 'physics_steps_saved': len(times),
                  'max_abs_recorded_control_nNm': float(np.max(np.abs(ctrl))),
                  'pedicel_angle_first_last_rad': positions, 'largest_accelerations': rows[:12], 'timeline': timeline,
                  'note': 'Read saved states only; reference inertia and partial trajectory do not prove a single causal mechanism.'}
        warned_joints = {item.get('joint') for item in failure['warnings']}
        result['warned_joint_history'] = {}
        for joint in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if name in warned_joints and int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                address, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
                result['warned_joint_history'][name] = {
                    'angle_first_last_rad': [float(qpos[0, address]), float(qpos[-1, address])],
                    'max_abs_velocity_rad_s': float(np.max(np.abs(velocity[:, dof]))),
                    'max_abs_acceleration_rad_s2': float(np.max(np.abs(acceleration[:, dof]))),
                    'inertia_at_reference_g_mm2': float(model.dof_M0[dof]),
                    'actuator_count': int(np.count_nonzero(model.actuator_trnid[:, 0] == joint)),
                }
    (run / 'failure_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    compact = {'failure': failure, 'physics_steps_saved': result['physics_steps_saved'],
               'max_abs_recorded_control_nNm': result['max_abs_recorded_control_nNm'],
               'warned_joint_history': result['warned_joint_history'], 'largest_accelerations': rows[:4],
               'timeline': [{key: value for key, value in sample.items() if key != 'contact_pairs'} |
                            {'contact_count': len(sample['contact_pairs'])} for sample in timeline]}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    analyze(parser.parse_args().run)
