"""Analyze saved fly-body data only. Does not step MuJoCo or the CNS."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
from inspect_recorded_run import events, recorded_state


def analyze(run):
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('output_mode') != 'public_fly_body':
        raise ValueError('새 초파리 몸체(full_fly_*) 기록을 선택하세요.')
    log = events(run)
    if not log:
        raise ValueError('완료된 기록 프레임이 없습니다.')
    output = run / 'fly_analysis'
    output.mkdir(exist_ok=True)
    annotation = pd.read_feather(run / 'annotations.feather')
    neural_indices = annotation.graph_index.to_numpy(dtype=np.int64)
    activities, rows, joint_rows, foot_rows = [], [], [], []
    for event in log:
        body, outputs = event['body'], event['motor_outputs']
        with recorded_state(run / event['snapshot']) as (snapshot, at):
            activities.append(at(neural_indices))
            # Raw camera is retained in each NPZ; small PPM previews are convenient for review.
            if event['frame'] in (0, log[len(log)//2]['frame'], log[-1]['frame']):
                rgb = snapshot['rgb']
                (output / f'camera_{event["frame"]:07d}.ppm').write_bytes(
                    f'P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n'.encode() + rgb.tobytes())
            physics_times = snapshot['physics_time_s']
            positions = snapshot['physics_body_xpos_mm'][:, metadata['body_root_id'], :]
            xy_steps = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
            physics_path = float(xy_steps.sum())
            warnings = snapshot['physics_warning_counts'][-1].tolist()
            if len(physics_times) != metadata['physics_steps_per_neural_tick']:
                raise ValueError('물리 단계 기록 수가 실행 metadata와 다릅니다.')
        rows.append({'frame': event['frame'], 'episode': event['episode'], 'case': event['case'],
                     'tick': event['neural_tick'], 'sim_seconds': event['sim_seconds'],
                     'stimulus_active': event['stimulus']['active'],
                     'body_x_mm': body['xyz_mm'][0], 'body_y_mm': body['xyz_mm'][1], 'body_z_mm': body['xyz_mm'][2],
                     'forward_velocity_mm_s': body['forward_velocity_mm_s'],
                     'yaw_velocity_world_rad_s': body['angular_velocity_world_rad_s'][2],
                     'upright_cosine': body['up_axis_world'][2], 'path_mm': body['path_mm'],
                     'within_frame_path_mm': physics_path,
                     'grounded_legs': sum(v['ground_contact'] for v in event['feet'].values()),
                     'max_abs_torque_nNm': max(abs(t) for t in outputs['torque_nNm']),
                     'max_activation': max(outputs['activation']),
                     'warning_counts': json.dumps(warnings), 'neural_compute_ms': event['neural_compute_ms'],
                     'save_ms': event['save_ms']})
        for i, name in enumerate(outputs['joint_names']):
            joint_rows.append({'frame': event['frame'], 'episode': event['episode'], 'case': event['case'],
                               'sim_seconds': event['sim_seconds'], 'joint': name,
                               'angle_deg': np.degrees(outputs['angle_rad'][i]),
                               'velocity_deg_s': np.degrees(outputs['velocity_rad_s'][i]),
                               'torque_nNm': outputs['torque_nNm'][i]})
        for leg, info in event['feet'].items():
            foot_rows.append({'frame': event['frame'], 'episode': event['episode'], 'case': event['case'],
                              'leg': leg, 'ground_contact': info['ground_contact'],
                              'x_mm': info['xyz_mm'][0], 'y_mm': info['xyz_mm'][1], 'z_mm': info['xyz_mm'][2]})
    timeline = pd.DataFrame(rows)
    timeline.to_csv(output / 'body_timeline.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(joint_rows).to_csv(output / 'joint_timeline.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(foot_rows).to_csv(output / 'foot_timeline.csv', index=False, encoding='utf-8-sig')
    activities = np.asarray(activities, dtype=np.float64)
    features = timeline[['forward_velocity_mm_s', 'yaw_velocity_world_rad_s', 'grounded_legs']].to_numpy(dtype=float)
    # Center each reset episode independently; no joining the ends of distinct trials.
    centered_x, centered_y = np.zeros_like(activities), np.zeros_like(features)
    eligible_frames = np.zeros(len(log), dtype=bool)
    for episode in timeline.episode.unique():
        mask = timeline.episode.eq(episode).to_numpy() & timeline.stimulus_active.to_numpy(dtype=bool)
        if mask.sum() >= 3:
            centered_x[mask] = activities[mask]-activities[mask].mean(axis=0)
            centered_y[mask] = features[mask]-features[mask].mean(axis=0)
            eligible_frames |= mask
    energy_x = np.square(centered_x).sum(axis=0)
    energy_y = np.square(centered_y).sum(axis=0)
    numerator = centered_x.T @ centered_y
    denominator = np.sqrt(energy_x[:, None] * energy_y[None, :])
    correlation = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=correlation, where=denominator > 1e-24)
    ranking = annotation.copy()
    ranking['mean_activity'] = activities.mean(axis=0)
    ranking['max_activity'] = activities.max(axis=0)
    for i, label in enumerate(('forward', 'yaw', 'contact')):
        ranking['correlation_' + label] = correlation[:, i]
    score = np.max(np.where(np.isfinite(correlation), np.abs(correlation), -1), axis=1)
    ranking['correlation_score'] = np.where(score >= 0, score, np.nan)
    ranking.sort_values('correlation_score', ascending=False, na_position='last').to_csv(
        output / 'annotated_motion_candidates.csv', index=False, encoding='utf-8-sig')
    episodes = []
    by_frame = {event['frame']: event for event in log}
    for episode, group in timeline.groupby('episode', sort=False):
        first, last = group.iloc[0], group.iloc[-1]
        episodes.append({'episode': int(episode), 'case': first['case'], 'frames': len(group),
                         'sim_seconds': float(last.sim_seconds), 'path_mm': float(last.path_mm),
                         'displacement_xy_mm': float(np.hypot(last.body_x_mm-by_frame[int(first['frame'])]['input_body']['xyz_mm'][0],
                                                              last.body_y_mm-by_frame[int(first['frame'])]['input_body']['xyz_mm'][1])),
                         'max_abs_forward_velocity_mm_s': float(group.forward_velocity_mm_s.abs().max()),
                         'min_upright_cosine': float(group.upright_cosine.min()),
                         'max_abs_joint_torque_nNm': float(group.max_abs_torque_nNm.max()),
                         'stimulated_frames': int(group.stimulus_active.sum())})
    report = {'run': str(run), 'frames': len(log), 'episodes': episodes,
              'annotated_graph_ids_ranked': len(annotation), 'correlation_frames': int(eligible_frames.sum()),
              'no_variation_features': [name for i, name in enumerate(('forward', 'yaw', 'contact')) if energy_y[i] <= 1e-24],
              'whole_graph_ranking': 'Use analyze_full_cns.py for stimulus contrasts over all annotated and unannotated graph IDs.',
              'mapping': {'mapped_mn': metadata['muscle_decoder']['mapped_mn_count'],
                          'unmapped_mn': metadata['muscle_decoder']['unmapped_mn_count']},
              'limitations': ['Correlations are candidates, not causal proof.',
                             'Constant input, time, body motion, gravity and the prescribed motor mapping confound interpretation.',
                             'No variation or fewer than three post-onset frames gives undefined correlation, never zero evidence.',
                             'No proprioceptive feedback was injected. Motor output coverage is incomplete.']}
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 전체 CNS · 초파리 몸체 기록 분석', '',
             f'완료 프레임 {len(log)}개. 물리 계산과 신경망을 재실행하지 않고 저장된 기록만 읽었습니다.', '',
             '| 에피소드 | 조건 | 가상 시간 s | 경로 mm | 순이동 mm | 최대 전진축 속도 mm/s |',
             '|---|---|---:|---:|---:|---:|']
    for item in episodes:
        lines.append(f'| {item["episode"]} | {item["case"]} | {item["sim_seconds"]:.3f} | {item["path_mm"]:.6f} | {item["displacement_xy_mm"]:.6f} | {item["max_abs_forward_velocity_mm_s"]:.6f} |')
    lines += ['', '경로는 중력으로 밀리거나 쓰러진 이동도 포함합니다. 전진 보행 판정은 발 접촉·관절 운동·몸체 방향을 함께 확인해야 합니다.',
              '', f'관절 출력 대응 운동신경 {report["mapping"]["mapped_mn"]}개, 미대응 {report["mapping"]["unmapped_mn"]}개.',
              '', 'annotated_motion_candidates.csv는 모든 주석 뉴런의 활동과 전진축 속도·회전·접촉의 상관 후보입니다. 인과 관계나 전진 기능의 확정 결과가 아닙니다.',
              '미주석 segment를 포함한 전체 그래프의 자극 반응 순위는 analyze_full_cns.py로 산출할 수 있습니다.',
              '', 'body_timeline.csv, joint_timeline.csv, foot_timeline.csv와 원본 NPZ의 모든 물리 단계 기록을 함께 사용하세요.']
    (output / 'report_ko.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'기록 분석 저장: {output / "report_ko.md"}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    analyze(parser.parse_args().run)
