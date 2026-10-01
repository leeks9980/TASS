"""Editable named-MN readout. Moment signs are explicit hypotheses, not connectome facts."""
import json
import numpy as np
from prepare_fly_body import LEGS, DRIVEN_SUFFIXES

# Match muscle annotations, without inventing a muscle identity for anonymous MNs.
PATTERNS = {
    'promotor': r'tergopleural/pleural promotor', 'remotor': r'pleural remotor/abductor',
    'anterior_rotator': r'sternal anterior rotator', 'posterior_rotator': r'sternal posterior rotator',
    'adductor': r'sternal adductor', 'tr_flexor': r'\btr\s+flexor\b',
    'tr_extensor': r'\btr\s+extensor\b', 'tibia_flexor': r'\bti\s+flexor\b',
    'tibia_extensor': r'\bti\s+extensor\b', 'tarsus_depressor': r'\bta\s+depressor\b',
    'tarsus_levator': r'\bta\s+levator\b',
}
REGION = {'F': 'fl', 'M': 'ml', 'H': 'hl'}


def default_mapping():
    channels, moments = [], []
    for leg in LEGS:
        for function in PATTERNS:
            channels.append({'name': leg + ':' + function, 'leg': leg, 'function': function})
        # Biological function names identify the target class, but not the exact 3-D moment arm.
        # These constant moment hypotheses can be replaced by measured tendon paths later.
        recipes = {
            'Coxa': [('promotor', -1), ('remotor', 1)],
            'Coxa_yaw': [('remotor', 1), ('adductor', -1)],
            'Coxa_roll': [('anterior_rotator', 1), ('posterior_rotator', -1)],
            'Femur': [('tr_flexor', -1), ('tr_extensor', 1)],
            'Femur_roll': [],  # No reliable named counterpart: passive joint, no guessed MN.
            'Tibia': [('tibia_flexor', 1), ('tibia_extensor', -1)],
            'Tarsus1': [('tarsus_depressor', 1), ('tarsus_levator', -1)],
        }
        for suffix, recipe in recipes.items():
            # Opposite global axes on the two sides; still an explicit unvalidated adapter.
            mirror = -1 if leg.startswith('R') and suffix in ('Coxa_yaw', 'Coxa_roll') else 1
            for function, sign in recipe:
                moments.append({'channel': leg + ':' + function, 'joint': 'joint_' + leg + suffix,
                                'moment_mm': 0.05 * sign * mirror})
    return {'schema': 1, 'status': 'engineering hypotheses; not anatomically calibrated',
            'channels': channels, 'moments': moments,
            'sources': ['https://github.com/NeLy-EPFL/flygym/tree/v1.2.1',
                        'MaleCNS v1.0 annotated muscle-function names'],
            'limitations': ['Constant moment arms and axis signs are hypotheses.',
                            'Remotor affects two axes here; no measured 3-D muscle path available.',
                            'Femur roll and anonymous/accessory muscle groups without explicit mapping remain unactuated.',
                            'Missing front/middle/hind annotations never filled by contralateral or homolog substitution.']}


class FlyMotorDecoder:
    def __init__(self, annotation, mapping_path, activity_scale=1e-4, force_uN=40., tau=.1):
        self.mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
        self.channels = self.mapping['channels']
        self.names = [x['name'] for x in self.channels]
        self.joint_names = ['joint_' + leg + suffix for leg in LEGS for suffix in DRIVEN_SUFFIXES]
        self.activity_scale, self.force_uN, self.tau = activity_scale, force_uN, tau
        if len(set(self.names)) != len(self.names):
            raise ValueError('운동 출력 채널 이름이 중복되었습니다.')
        self.moments = np.zeros((len(self.joint_names), len(self.names)))
        for item in self.mapping['moments']:
            if not np.isfinite(item['moment_mm']):
                raise ValueError('모멘트암은 유한한 값이어야 합니다.')
            self.moments[self.joint_names.index(item['joint']), self.names.index(item['channel'])] += item['moment_mm']
        sides = annotation.somaSide.fillna(annotation.rootSide).fillna('')
        types = annotation.type.fillna('').str.lower()
        motor = annotation.superclass.eq('vnc_motor') & annotation.subclass.isin(list(REGION.values()))
        self.groups, membership, mapped = [], {}, set()
        for channel in self.channels:
            leg, fn = channel['leg'], channel['function']
            mask = motor & sides.eq(leg[0]) & annotation.subclass.eq(REGION[leg[1]])
            if 'body_ids' in channel:
                # Explicit target evidence can describe a crossed projection; do not force soma-side.
                mask = motor & annotation.bodyId.isin(channel['body_ids'])
            else:
                pattern = channel.get('annotation_regex', PATTERNS.get(fn))
                if not pattern:
                    raise ValueError(f'운동신경 대응 근거가 없습니다: {channel["name"]}')
                mask &= types.str.contains(pattern, regex=True)
            selected = annotation.loc[mask]
            self.groups.append(selected.graph_index.to_numpy(dtype=np.int64))
            membership[channel['name']] = selected.bodyId.astype('int64').tolist()
            # Count only channels that actually have a mechanical moment.
            if np.any(self.moments[:, len(self.groups)-1]):
                mapped.update(membership[channel['name']])
        self.unmapped = annotation.loc[motor & ~annotation.bodyId.isin(mapped)].copy()
        self.all_motor = annotation.loc[motor].copy()
        self.report = {'mapping': self.mapping, 'membership_body_ids': membership,
                       'joint_order': self.joint_names, 'channel_order': self.names,
                       'leg_mn_count': int(motor.sum()), 'mapped_mn_count': len(mapped),
                       'unmapped_mn_count': len(self.unmapped),
                       'missing_channels': [n for n, g in zip(self.names, self.groups) if not len(g)],
                       'activity_scale': activity_scale, 'max_force_uN': force_uN, 'activation_tau_s': tau,
                       'equation': 'mean(state[group]) -> clip(raw / scale,0,1) -> first-order activation -> force_uN; torque_nNm = moment_mm @ force_uN',
                       'physiology': 'No Hill force-length/velocity, receptor models, tendon optimization or learned controller.',
                       'side_basis': 'somaSide falling back to rootSide, assumed target side; crossed motor targets not reconstructed'}
        self.reset()

    def reset(self):
        self.raw = np.zeros(len(self.channels))
        self.targets = np.zeros_like(self.raw)
        self.activation = np.zeros_like(self.raw)
        self.forces = np.zeros_like(self.raw)
        self.torques = np.zeros(len(self.joint_names))

    def observe(self, state):
        self.raw[:] = [float(state[g].mean()) if len(g) else 0. for g in self.groups]
        self.targets[:] = np.clip(self.raw / self.activity_scale, 0., 1.)

    def advance(self, dt):
        self.activation += -np.expm1(-dt / self.tau) * (self.targets - self.activation)
        self.forces[:] = self.force_uN * self.activation
        self.torques[:] = self.moments @ self.forces
        return self.torques

    def snapshot(self, angles, velocities):
        return {'channel_names': self.names, 'raw_activity': self.raw.tolist(),
                'target_activation': self.targets.tolist(), 'activation': self.activation.tolist(),
                'force_uN': self.forces.tolist(), 'joint_names': self.joint_names,
                'torque_nNm': self.torques.tolist(), 'angle_rad': angles.tolist(),
                'velocity_rad_s': velocities.tolist()}
