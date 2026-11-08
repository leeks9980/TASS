from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from experiment_config import DEFAULT_EXPERIMENT, load_experiment
from full_cns_model import FLYNN_EQUATION, flynn_leaky_update
from safe_dynamics import DynamicsExpression


def test_flynn_update_preserves_signed_tanh_state():
    state = np.zeros(2, dtype=np.float32)
    drive = np.array([1.0, -1.0], dtype=np.float32)

    flynn_leaky_update(state, drive, retention=0.65)

    expected = 0.35 * np.tanh(np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(state, expected)
    assert state[0] > 0
    assert state[1] < 0


def test_inhibitory_second_hop_is_not_clipped_to_zero():
    state = np.zeros(2, dtype=np.float32)

    first_drive = np.array([1.0, 0.0], dtype=np.float32)
    flynn_leaky_update(state, first_drive, retention=0.65)
    assert state[0] > 0
    assert state[1] == 0

    # Neuron 0 inhibits neuron 1 on the next recurrent tick.
    second_drive = np.array([0.0, -state[0]], dtype=np.float32)
    flynn_leaky_update(state, second_drive, retention=0.65)

    assert state[1] < 0


def test_editable_flynn_expression_matches_reference_update():
    h = np.array([0.2, -0.3], dtype=np.float32)
    recurrent = np.array([-0.4, 0.5], dtype=np.float32)
    sensory = np.array([0.1, 0.0], dtype=np.float32)
    expression = DynamicsExpression(FLYNN_EQUATION)

    result = expression.evaluate(h=h, recurrent=recurrent, sensory=sensory,
                                 gain=0.95, leak=0.65, bias=0.0)
    expected = 0.65*h + 0.35*np.tanh(0.95*recurrent + sensory)

    np.testing.assert_allclose(result, expected)


def test_editable_legacy_expression_can_be_selected_for_comparison():
    expression = DynamicsExpression(
        'leak*h + (1-leak)*maximum(0, tanh(gain*recurrent + sensory + bias))')
    result = expression.evaluate(
        h=np.zeros(1, dtype=np.float32), recurrent=np.array([-1.], dtype=np.float32),
        sensory=np.zeros(1, dtype=np.float32), gain=1., leak=.65, bias=0.)

    np.testing.assert_array_equal(result, np.zeros(1, dtype=np.float32))


def test_expression_rejects_python_attributes_and_unknown_names():
    for invalid in ['np.tanh(h)', '__import__("os")', 'unknown + h', 'h[0]']:
        try:
            DynamicsExpression(invalid)
        except ValueError:
            continue
        raise AssertionError(f'unsafe expression accepted: {invalid}')


def test_default_experiment_file_loads_all_runtime_defaults():
    experiment = load_experiment(DEFAULT_EXPERIMENT)

    assert experiment['document']['name'] == 'flynn_signed_visual_motor_corridor_tuned'
    assert experiment['defaults']['equation'] == FLYNN_EQUATION
    assert experiment['defaults']['auto_protocol'] is True
    assert experiment['defaults']['require_corridor'] is True
