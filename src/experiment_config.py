"""Load the single editable experiment configuration file."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT = ROOT / 'experiment.json'


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_experiment(path=DEFAULT_EXPERIMENT):
    path = _resolve(path).resolve()
    raw = path.read_bytes()
    document = json.loads(raw.decode('utf-8'))
    if document.get('schema') != 1:
        raise ValueError('experiment.json schema must be 1')

    required = ['paths', 'dynamics', 'motor', 'simulation', 'protocol']
    missing = [key for key in required if not isinstance(document.get(key), dict)]
    if missing:
        raise ValueError(f'experiment.json sections missing: {missing}')

    paths = document['paths']
    dynamics = document['dynamics']
    motor = document['motor']
    simulation = document['simulation']
    protocol = document['protocol']
    defaults = {
        'circuit': _resolve(paths['circuit']),
        'mapping': _resolve(paths['motor_mapping']),
        'require_corridor': bool(paths['require_corridor']),
        'glutamate': dynamics['glutamate'],
        'weight_mode': dynamics['weight_mode'],
        'gain': dynamics['gain'],
        'leak': dynamics['leak'],
        'bias': dynamics['bias'],
        'equation': dynamics['equation'],
        'activity_scale': motor['activity_scale'],
        'muscle_force_uN': motor['muscle_force_uN'],
        'muscle_tau': motor['muscle_tau'],
        'max_joint_torque_nNm': motor['max_joint_torque_nNm'],
        'max_torque_rate_nNm_s': motor['max_torque_rate_nNm_s'],
        'motor_settle': motor['motor_settle'],
        'motor_ramp': motor['motor_ramp'],
        'actuation_mode': motor['actuation_mode'],
        'fixed_body': bool(simulation['fixed_body']),
        'neural_interval': simulation['neural_interval'],
        'record_gb': simulation['record_gb'],
        'case': protocol['manual_case'],
        'stimulus_onset': protocol['stimulus_onset'],
        'auto_protocol': bool(protocol['enabled']),
        'auto_cases': protocol['cases'],
        'auto_case_seconds': protocol['case_seconds'],
    }
    return {
        'path': path,
        'sha256': hashlib.sha256(raw).hexdigest(),
        'document': document,
        'defaults': defaults,
    }
