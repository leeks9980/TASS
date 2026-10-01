"""Summarize existing frames and failure data for interpretation. Never simulate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import mujoco
import numpy as np


def explain(run):
    model = mujoco.MjModel.from_binary_path(str(run / 'body_model.mjb'))
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    log = [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    summaries = []
    names = metadata['model_names']['geoms']
    for event in log:
        contact_names = sorted({names[index] for item in event['contacts'] for index in item['geom_ids']
                                if metadata['floor_geom_id'] in item['geom_ids'] and index != metadata['floor_geom_id']})
        outputs = event['motor_outputs']
        with np.load(run / event['snapshot']) as snapshot:
            velocity = snapshot['physics_qvel']
            acceleration = snapshot['physics_qacc']
            sample, dof = np.unravel_index(np.argmax(np.abs(velocity)), velocity.shape)
            joint = int(model.dof_jntid[dof])
        summaries.append({'frame': event['frame'], 'sim_seconds': event['sim_seconds'],
                          'stimulus_active': event['stimulus']['active'],
                          'body_height_mm': event['body']['xyz_mm'][2], 'path_mm': event['body']['path_mm'],
                          'forward_velocity_mm_s': event['body']['forward_velocity_mm_s'],
                          'max_abs_control_nNm': float(np.max(np.abs(outputs['torque_nNm']))),
                          'MN_L': event['motor_scores']['L'], 'MN_R': event['motor_scores']['R'],
                          'max_abs_physics_qvel': float(np.max(np.abs(velocity))),
                          'max_velocity_joint': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                          'max_abs_physics_qacc': float(np.max(np.abs(acceleration))),
                          'nonzero_graph_states': event['nonzero_states'],
                          'thorax_on_floor': 'Thorax' in contact_names,
                          'floor_contact_parts': contact_names})
    if not summaries:
        raise ValueError('완료된 프레임이 없습니다.')
    result = {'initial_body': log[0]['input_body'], 'frames': summaries,
              'total_body_mass_g': float(model.body_mass.sum()),
              'model_weight_uN': float(model.body_mass.sum()*abs(model.opt.gravity[2])),
              'failure': json.loads((run / 'failure.json').read_text(encoding='utf-8')) if (run / 'failure.json').exists() else None,
              'reading': 'Saved physical states and compiled parameters only; no neural update or physical integration.'}
    (run / 'explanation_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    output = result | {'frames': [{k: v for k, v in frame.items() if k != 'floor_contact_parts'} for frame in summaries]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    explain(parser.parse_args().run)
