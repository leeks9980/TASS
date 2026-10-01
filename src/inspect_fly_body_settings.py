"""Read an existing compiled body and its settings, without stepping physics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import mujoco
import numpy as np


def inspect(run):
    model = mujoco.MjModel.from_binary_path(str(run / 'body_model.mjb'))
    rows = []
    for joint in range(model.njnt):
        if int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        dof = int(model.jnt_dofadr[joint])
        k = float(model.jnt_stiffness[joint])
        inertia = float(model.dof_M0[dof])
        inverse_inertia = float(model.dof_invweight0[dof])
        rows.append({'joint': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                     'stiffness_nNm_per_rad': k, 'damping_nNm_s_per_rad': float(model.dof_damping[dof]),
                     'inertia_at_qpos0_g_mm2': inertia, 'inverse_inertia_at_qpos0': inverse_inertia,
                     'dt_sqrt_k_inverse_inertia': float(model.opt.timestep*np.sqrt(k*inverse_inertia))})
    rows.sort(key=lambda item: item['dt_sqrt_k_inverse_inertia'], reverse=True)
    result = {'mujoco_version': mujoco.__version__, 'timestep_s': float(model.opt.timestep),
              'note': 'Read-only inspection at compiled reference pose; not a simulation or causal diagnosis.',
              'joints': rows}
    (run / 'body_settings_inspection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'timestep_s': result['timestep_s'], 'largest_reference_ratios': rows[:8]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    inspect(parser.parse_args().run)
