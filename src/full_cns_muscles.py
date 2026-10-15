"""Explicit annotation-based muscle readout; no network pruning or action policy."""
import numpy as np


LEGS = ('L_fl', 'L_ml', 'L_hl', 'R_fl', 'R_ml', 'R_hl')
LEG_LABELS = {'L_fl': '왼쪽 앞', 'L_ml': '왼쪽 중간', 'L_hl': '왼쪽 뒤',
              'R_fl': '오른쪽 앞', 'R_ml': '오른쪽 중간', 'R_hl': '오른쪽 뒤'}


class MuscleDecoder:
    """Two antagonists per tibia joint; force coefficients are engineering assumptions."""

    def __init__(self, annotation, activity_scale=1e-4, force_max=4.,
                 moment_arm=.04, time_constant=.1):
        self.activity_scale, self.force_max = activity_scale, force_max
        self.moment_arm, self.time_constant = moment_arm, time_constant
        sides = annotation.somaSide.fillna(annotation.rootSide).fillna('')
        types = annotation.type.fillna('').str.lower()
        motor = annotation.superclass.eq('vnc_motor') & annotation.subclass.isin(['fl', 'ml', 'hl'])
        self.groups, self.membership, mapped = {}, {}, set()
        for leg in LEGS:
            side, region = leg.split('_')
            self.groups[leg], self.membership[leg] = {}, {}
            for function in ('flexor', 'extensor'):
                # Match named tibia MNs, including annotated accessory tibia flexors.
                named = types.str.contains(r'\b(?:ti|tibia)\s+' + function + r'\b', regex=True)
                selected = annotation.loc[motor & sides.eq(side) & annotation.subclass.eq(region) & named]
                self.groups[leg][function] = selected.graph_index.to_numpy(dtype=np.int64)
                bodies = selected.bodyId.astype('int64').tolist()
                self.membership[leg][function] = bodies
                mapped.update(bodies)
        self.steering = {}
        for name in ('DNa02', 'DNg13'):
            self.steering[name] = {
                side: annotation.loc[annotation.type.eq(name) & sides.eq(side), 'graph_index'].to_numpy(dtype=np.int64)
                for side in ('L', 'R')}
        self.report = {
            'output_mode': 'antagonistic_tibia_muscles', 'joint_order': list(LEGS),
            'membership_body_ids': self.membership,
            'readout_leg_mn_count': len(mapped),
            'other_leg_mn_count_retained_in_network': int(motor.sum())-len(mapped),
            'missing_groups': [f'{leg}:{fn}' for leg in LEGS for fn, ids in self.groups[leg].items() if not len(ids)],
            'pooling': 'equal mean within annotated tibia muscle function; uncalibrated',
            'normalization': 'clip(mean_activity / fixed_activity_scale, 0, 1); no per-frame rescaling',
            'parameters': {'activity_scale': activity_scale, 'force_max_N': force_max,
                           'moment_arm_m': moment_arm, 'activation_time_constant_s': time_constant},
            'torque_equation': 'moment_arm * (flexor_force - extensor_force)',
            'muscle_force_equation': 'force_max * activation; no force-length/velocity physiology',
            'steering': 'DNa02/DNg13 soma-side diagnostics only; crossed axons not interpreted as motor side',
            'body': 'fixed six-joint diagnostic rig, zero gravity; not a walking or two-motor robot model',
        }
        self.reset()

    def reset(self):
        self.raw = np.zeros((len(LEGS), 2), dtype=float)
        self.targets = np.zeros_like(self.raw)
        self.activation = np.zeros_like(self.raw)
        self.forces = np.zeros_like(self.raw)
        self.torques = np.zeros(len(LEGS), dtype=float)
        self.steering_values = {}

    def observe(self, state):
        for i, leg in enumerate(LEGS):
            for j, function in enumerate(('flexor', 'extensor')):
                ids = self.groups[leg][function]
                self.raw[i, j] = float(state[ids].mean()) if len(ids) else 0.
        self.targets = np.clip(self.raw / self.activity_scale, 0., 1.)
        self.steering_values = {}
        for name, groups in self.steering.items():
            values = {side: float(state[ids].mean()) if len(ids) else None for side, ids in groups.items()}
            values['R_minus_L'] = values['R']-values['L'] if all(values[s] is not None for s in ('L', 'R')) else None
            self.steering_values[name] = values

    def advance(self, dt):
        # Exact first-order activation interpolation, advanced at each physics step.
        alpha = -np.expm1(-dt / self.time_constant)
        self.activation += alpha * (self.targets-self.activation)
        self.forces = self.force_max * self.activation
        self.torques = self.moment_arm * (self.forces[:, 0]-self.forces[:, 1])
        return self.torques.copy()

    def snapshot(self, angles, velocities):
        return {leg: {
            'flexor_activity': float(self.raw[i, 0]), 'extensor_activity': float(self.raw[i, 1]),
            'flexor_target': float(self.targets[i, 0]), 'extensor_target': float(self.targets[i, 1]),
            'flexor_activation': float(self.activation[i, 0]), 'extensor_activation': float(self.activation[i, 1]),
            'flexor_force_N': float(self.forces[i, 0]), 'extensor_force_N': float(self.forces[i, 1]),
            'torque_Nm': float(self.torques[i]), 'angle_rad': float(angles[i]),
            'angular_velocity_rad_s': float(velocities[i]),
            'flexor_count': len(self.groups[leg]['flexor']), 'extensor_count': len(self.groups[leg]['extensor']),
        } for i, leg in enumerate(LEGS)}


def compare_to_neutral(current, reference):
    """Diagnostic comparison at the same neural tick; never controls the muscles."""
    if reference is None:
        return None
    fields = ('flexor_activity', 'extensor_activity', 'flexor_force_N', 'extensor_force_N', 'torque_Nm', 'angle_rad')
    return {leg: {field: current[leg][field]-reference[leg][field] for field in fields} for leg in LEGS}
