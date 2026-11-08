"""Validate and summarize an experiment JSON without starting MuJoCo."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np

from experiment_config import DEFAULT_EXPERIMENT, load_experiment
from safe_dynamics import DynamicsExpression


def check(path):
    experiment = load_experiment(path)
    defaults = experiment['defaults']
    expression = DynamicsExpression(defaults['equation'])
    sample = np.array([0.2, -0.3, 0.0], dtype=np.float32)
    result = expression.evaluate(
        h=sample, recurrent=np.array([-0.4, 0.5, 0.1], dtype=np.float32),
        sensory=np.array([0.1, 0.0, 0.2], dtype=np.float32),
        gain=float(defaults['gain']), leak=float(defaults['leak']),
        bias=float(defaults['bias']))
    if not defaults['circuit'].joinpath('report.json').is_file():
        raise FileNotFoundError(f"circuit report not found: {defaults['circuit'] / 'report.json'}")
    if not defaults['mapping'].is_file():
        raise FileNotFoundError(f"motor mapping not found: {defaults['mapping']}")
    summary = {
        'valid': True,
        'name': experiment['document'].get('name'),
        'configuration': str(experiment['path']),
        'sha256': experiment['sha256'],
        'circuit': str(defaults['circuit']),
        'equation': expression.expression,
        'equation_sample_output': result.tolist(),
        'gain': defaults['gain'], 'leak': defaults['leak'], 'bias': defaults['bias'],
        'automatic_cases': defaults['auto_cases'] if defaults['auto_protocol'] else [],
        'case_seconds': defaults['auto_case_seconds'],
        'stimulus_onset': defaults['stimulus_onset'],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    try:
        check(args.experiment)
    except Exception as exc:
        print(f'실험 설정 오류: {exc}', file=sys.stderr)
        raise SystemExit(1)
