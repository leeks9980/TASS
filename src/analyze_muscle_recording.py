"""Audit saved muscle recordings and write a report; no simulation or learning."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from inspect_recorded_run import events, recorded_state

LABELS = {'L_fl': '왼쪽 앞', 'L_ml': '왼쪽 중간', 'L_hl': '왼쪽 뒤',
          'R_fl': '오른쪽 앞', 'R_ml': '오른쪽 중간', 'R_hl': '오른쪽 뒤'}


def analyze(run):
    meta = json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('output_mode') != 'antagonistic_tibia_muscles':
        raise ValueError('Muscle-model recording required')
    log = events(run)
    if not log:
        raise ValueError('No committed frames')
    decoder = meta['muscle_decoder']
    legs = decoder['joint_order']
    parameters = decoder['parameters']
    annotation = feather.read_feather(run/'annotations.feather')
    graph_for_body = dict(zip(annotation.bodyId, annotation.graph_index))
    membership = decoder['membership_body_ids']
    group_indices = {(leg, fn): np.array([graph_for_body[body] for body in membership[leg][fn]], dtype=np.int64)
                     for leg in legs for fn in ('flexor', 'extensor')}
    # This XML is the current rig configuration, not an archived asset snapshot.
    arena = ET.parse(ROOT/meta['arena']).getroot()
    joint_default = arena.find('default/joint')
    lower_deg, upper_deg = map(float, joint_default.attrib['range'].split())
    stiffness = float(joint_default.attrib['stiffness'])
    dt = float(arena.find('option').attrib['timestep'])
    rows, audit_rows, previous = [], [], {}
    max_errors = {key: 0. for key in ('raw_activity', 'target', 'activation', 'force', 'torque', 'angle_vs_qpos', 'torque_vs_command')}
    for event in log:
        with recorded_state(run/event['snapshot']) as (_, at):
            for i, leg in enumerate(legs):
                item = event['muscles'][leg]
                row = {'run': run.name, 'frame': event['frame'], 'episode': event['episode'], 'case': event['case'],
                       'sim_seconds': event['sim_seconds'], 'input_sim_seconds': event['input_sim_seconds'],
                       'neural_tick': event['neural_tick'], 'leg': leg, **item}
                row['angle_deg'] = float(np.degrees(item['angle_rad']))
                row['velocity_deg_s'] = float(np.degrees(item['angular_velocity_rad_s']))
                row['near_lower_limit'] = row['angle_deg'] <= lower_deg+.2
                row['spring_only_equilibrium_deg'] = float(np.degrees(item['torque_Nm']/stiffness))
                row['lower_limit_excess_drive_Nm'] = max(0., -item['torque_Nm']-stiffness*abs(np.radians(lower_deg)))
                row['comparison_available'] = event.get('neutral_difference') is not None
                for fn in ('flexor', 'extensor'):
                    ids = group_indices[leg, fn]
                    raw = float(at(ids).mean()) if len(ids) else 0.
                    target = float(np.clip(raw/parameters['activity_scale'], 0, 1))
                    key = (event['episode'], leg, fn)
                    old = previous.get(key)
                    interval = event['sim_seconds']-(old[0] if old else 0.)
                    activation = target + ((old[1] if old else 0.)-target)*np.exp(-interval/parameters['activation_time_constant_s'])
                    force = parameters['force_max_N']*item[fn+'_activation']
                    checks = {'raw_activity': abs(raw-item[fn+'_activity']),
                              'target': abs(target-item[fn+'_target']),
                              'activation': abs(activation-item[fn+'_activation']),
                              'force': abs(force-item[fn+'_force_N'])}
                    for name, error in checks.items():
                        max_errors[name] = max(max_errors[name], error)
                    previous[key] = (event['sim_seconds'], item[fn+'_activation'])
                    audit_rows.append({'frame': event['frame'], 'episode': event['episode'], 'leg': leg,
                                       'function': fn, 'state_mean': raw, **checks})
                torque = parameters['moment_arm_m']*(item['flexor_force_N']-item['extensor_force_N'])
                max_errors['torque'] = max(max_errors['torque'], abs(torque-item['torque_Nm']))
                max_errors['angle_vs_qpos'] = max(max_errors['angle_vs_qpos'], abs(event['qpos'][i]-item['angle_rad']))
                max_errors['torque_vs_command'] = max(max_errors['torque_vs_command'], abs(event['motor_commands'][i]-item['torque_Nm']))
                rows.append(row)
    table = pd.DataFrame(rows)
    out = run/'diagnosis'
    out.mkdir(exist_ok=True)
    table.to_csv(out/'muscle_timeline.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(audit_rows).to_csv(out/'muscle_record_audit.csv', index=False, encoding='utf-8-sig')
    episodes = []
    for (episode, case), subset in table.groupby(['episode', 'case'], sort=False):
        final_rows = subset[subset.frame == subset.frame.max()]
        group = [e for e in log if e['episode'] == episode]
        last = group[-1]
        details = []
        for leg in legs:
            selected = subset[subset.leg == leg]
            final = final_rows[final_rows.leg == leg].iloc[0]
            near = selected[selected.near_lower_limit]
            details.append({'leg': leg, 'label': LABELS.get(leg, leg),
                            'first_near_lower_limit_s': float(near.sim_seconds.iloc[0]) if len(near) else None,
                            'final_angle_deg': float(final.angle_deg),
                            'final_velocity_deg_s': float(final.velocity_deg_s),
                            'final_torque_Nm': float(final.torque_Nm),
                            'final_flexor_force_N': float(final.flexor_force_N),
                            'final_extensor_force_N': float(final.extensor_force_N),
                            'min_angle_deg': float(selected.angle_deg.min()),
                            'max_angle_deg': float(selected.angle_deg.max()),
                            'post_1s_angle_span_deg': float(selected.loc[selected.sim_seconds >= 1., 'angle_deg'].max()-selected.loc[selected.sim_seconds >= 1., 'angle_deg'].min()) if (selected.sim_seconds >= 1.).any() else None,
                            'final_spring_only_equilibrium_deg': float(final.spring_only_equilibrium_deg),
                            'final_lower_limit_excess_drive_Nm': float(final.lower_limit_excess_drive_Nm)})
        episodes.append({'episode': int(episode), 'case': case, 'frames': len(group),
                         'sim_seconds': last['sim_seconds'], 'final_input_seconds': last['input_sim_seconds'],
                         'brightness_range': [min(e['stimulus']['light_multiplier'] for e in group), max(e['stimulus']['light_multiplier'] for e in group)],
                         'neutral_reference_frames': sum(e.get('neutral_difference') is not None for e in group),
                         'legs': details, 'final_steering': last['steering_diagnostics']})
    final = log[-1]
    compute = np.array([e['compute_ms_excluding_record'] for e in log])
    save = np.array([e['save_ms'] for e in log])
    snapshot_bytes = sum((run/e['snapshot']).stat().st_size for e in log)
    summary = {'run': run.name, 'frames': len(log), 'episodes': episodes,
               'max_record_reconstruction_errors': max_errors,
               'comparison_frames': sum(e.get('neutral_difference') is not None for e in log),
               'median_compute_seconds': float(np.median(compute)/1000),
               'median_save_seconds': float(np.median(save)/1000),
               'recorded_compute_and_save_seconds': float((compute+save).sum()/1000),
               'snapshot_bytes': snapshot_bytes, 'record_budget_GiB': meta['parameters']['record_gb'],
               'budget_reached_by_recorded_bytes': snapshot_bytes >= meta['parameters']['record_gb']*1024**3,
               'rig_reference': {'file': meta['arena'], 'current_lower_deg': lower_deg, 'current_upper_deg': upper_deg,
                                 'current_stiffness_Nm_per_rad': stiffness, 'current_timestep': dt,
                                 'note': 'Current XML is not an archived snapshot of the executed asset.'},
               'neural_activities_have_no_physiological_units': True,
               'readout_leg_motor_neurons': decoder['readout_leg_mn_count'],
               'other_leg_motor_neurons_retained': decoder['other_leg_mn_count_retained_in_network']}
    (out/'muscle_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 가상 근육·관절 기록 분석', '',
             f'분석 대상: `{run.name}`, 저장된 {len(log)}프레임. 새 시뮬레이션·신경망 진행·학습 없이 기록만 조회했습니다.', '',
             '현재 테스트는 카메라의 시각 자극 → 모델 신경 활동 → 가상 근육 힘 → 관절 토크·각도의 전달을 관찰하는 용도입니다. 몸체는 고정돼 있고 중력·접촉 감각 입력이 없는 진단 장치이므로 이동·자율 회피·보행 성능의 검증은 아닙니다.', '',
             '## 기록된 조건과 움직임', '']
    for episode in episodes:
        lines += [f'에피소드 {episode["episode"]}: `{episode["case"]}`, {episode["frames"]}프레임, 가상 {episode["sim_seconds"]:.2f}초.', '',
                  '| 관절 | 마지막 각도 | 마지막 굽힘 힘 | 마지막 폄 힘 | 마지막 토크 | −20° 부근 도달 시점 |',
                  '|---|---:|---:|---:|---:|---:|']
        for item in episode['legs']:
            hit = '미도달' if item['first_near_lower_limit_s'] is None else f'{item["first_near_lower_limit_s"]:.2f} s'
            lines.append(f'| {item["label"]} | {item["final_angle_deg"]:.3f}° | {item["final_flexor_force_N"]:.3f} N | {item["final_extensor_force_N"]:.3f} N | {item["final_torque_Nm"]:+.5f} Nm | {hit} |')
        lines += ['', f'광원 배율 범위: {episode["brightness_range"]}. 같은 신경 업데이트 시점의 무물체 기준이 있는 프레임: {episode["neutral_reference_frames"]}.', '']
    neural_summary_path = out/'summary.json'
    if neural_summary_path.exists():
        neural = json.loads(neural_summary_path.read_text(encoding='utf-8'))
        if neural['frames_analyzed'] == len(log):
            lines += ['## 마지막 프레임의 주석 신경 집단', '',
                      '아래 활동값은 모델 내부 무차원 수치이며 실제 발화율이 아닙니다. 집단 평균을 단일한 직렬 경로의 감쇠율로 해석하지 않습니다.', '',
                      '| 집단 | 평균 활동 | 0보다 큰 활동 / 주석 수 |', '|---|---:|---:|']
            for name in ('L1', 'L2', 'L3', 'visual_projection', 'descending_neuron', 'vnc_intrinsic', 'leg_motor'):
                values = neural['final_group_stats'][name]
                lines.append(f'| {name} | {values["mean"]:.6g} | {values["active"]:,} / {values["n"]:,} |')
            lines += ['', f'운동신경 381개의 직전 입력으로 재구성한 마지막 상태와 기록의 최대 절대 오차는 {neural["mn_reconstruction_max_error"]:.6g}입니다.', '',
                      f'전체 모델에는 다리 운동신경 381개가 보존됩니다. 이번 근육 출력에 읽는 경골 굽힘/폄 신경은 {decoder["readout_leg_mn_count"]}개이고, 다른 기능의 {decoder["other_leg_mn_count_retained_in_network"]}개도 신경망과 전체 상태 기록에 남아 있습니다.', '']
    lines += ['## 움직임이 멈춰 보이는 원인', '',
              '이 기록의 마지막 시점에서 모든 관절은 폄 힘이 굽힘 힘보다 큰 음의 토크를 받습니다. 각도가 초기 0°에서 폄 방향으로 이동한 뒤 현재 XML의 하한 −20° 근처에 머뭅니다. 이전 바퀴 모델의 작은 속도 명령과 달리, 이번 모델에서는 근육 토크가 발생해도 관절 이동 범위에 걸려 추가 변화가 제한됩니다.', '',
              '관절은 부드러운 제한 조건으로 계산되므로 −20°보다 약간 작은 값도 나올 수 있습니다. 코드의 중립 자세와 폄 방향 가동 범위를 제가 설정한 결과이며 실제 파리의 회피 실패나 운동 능력 부재로 해석할 수 없습니다.', '',
              '신경 활동은 아직 가상 시간에 따라 증가하는 초기 과정입니다. 물체 없음 기준과 다른 자극 조건이 없으면 이 변화가 물체 이동에 특이적인 반응인지 분리할 수 없습니다.', '',
              '## 기록의 일치 확인', '',
              '저장된 전체 신경 상태에서 근육 그룹의 평균 활동을 다시 읽고, 고정 활동 기준, 기록된 활성의 시간 관계, 힘·토크식을 대조했습니다. 새로운 물리 궤적이나 신경 상태를 생성한 것은 아닙니다.', '']
    lines.extend(f'- {name}: 최대 절대 오차 {value:.6g}.' for name, value in max_errors.items())
    lines += ['', '## 시간과 저장 범위', '',
              f'계산 중앙값 {summary["median_compute_seconds"]:.3f}초/프레임, 저장 중앙값 {summary["median_save_seconds"]:.3f}초/프레임. 기록된 계산·저장 시간의 합은 {summary["recorded_compute_and_save_seconds"]:.2f}초이며 일시정지·사용자 대기는 포함하지 않습니다.', '',
              f'스냅샷 합계 {snapshot_bytes/1024**3:.3f} GiB, 저장 한도 {summary["record_budget_GiB"]:g} GiB. 기록 바이트 기준 한도 도달: {summary["budget_reached_by_recorded_bytes"]}.', '',
              '## 다음 관찰을 위한 수정 방향', '',
              '근육 신호가 두 방향 모두에서 관절 변화로 나타나도록 중립 자세·양방향 가동 범위·근육 힘 대 관절 복원력의 비율을 보정하는 것이 우선입니다. 단순히 힘을 더 키우면 한계에 더 강하게 밀릴 수 있습니다. 이번 분석에서는 시뮬레이션 코드를 수정하지 않았습니다.', '',
              '무물체 기준과 각 자극을 동일한 초기 상태·경과 시간에서 비교해야 선택적인 시각 반응을 조사할 수 있습니다. 기본 10 GiB 기록은 약 1.65초에 소진되어 2초 후 광원 변화 조건을 관찰하기에 부족합니다.', '',
              '## 결과 파일', '',
              '- `muscle_timeline.csv`: 프레임·관절별 활동·힘·토크·각도·속도.',
              '- `muscle_record_audit.csv`: 저장 신경 상태와 근육 기록의 수치 대조.',
              '- `muscle_summary.json`: 가동 한계 도달 시점, 조건, 계산 시간과 대조 오차.',
              '- `muscle_response.png`: 가상 시간별 관절 각도·근육 힘·토크 그래프(생성 가능한 환경일 때).', '',
              '가동 범위와 복원력은 실행 시점에 별도로 보관되지 않은 현재 XML을 참조했습니다. 두 기록의 실행 설정과 일치하는 파일이며, 기록에서 모든 관절이 같은 각도 근처에 머무는 사실은 직접 확인됩니다.', '']
    (out/'muscle_analysis_ko.md').write_text('\n'.join(lines), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
        selected = table[table.episode == final['episode']]
        colors = plt.cm.tab10(np.linspace(0, .9, len(legs)))
        for color, leg in zip(colors, legs):
            values = selected[selected.leg == leg]
            axes[0].plot(values.sim_seconds, values.angle_deg, label=leg, color=color)
            axes[1].plot(values.sim_seconds, values.extensor_force_N, color=color)
            axes[1].plot(values.sim_seconds, values.flexor_force_N, color=color, linestyle='--')
            axes[2].plot(values.sim_seconds, values.torque_Nm, color=color)
        axes[0].axhline(lower_deg, color='black', linestyle=':', label=f'lower limit {lower_deg:g} deg')
        axes[0].set_ylabel('Joint angle (deg)')
        axes[0].legend(ncol=4, fontsize=8)
        axes[1].set_ylabel('Muscle force (N)')
        axes[1].set_title('Solid: extensor; dashed: flexor', fontsize=10)
        axes[2].set_ylabel('Joint torque (Nm)')
        axes[2].set_xlabel('Simulated time (s)')
        for ax in axes:
            ax.grid(alpha=.3)
        fig.suptitle(f'{run.name} | recorded {final["case"]} episode {final["episode"]}')
        fig.savefig(out/'muscle_response.png', dpi=160)
        plt.close(fig)
    except ImportError:
        print('Plot dependency unavailable; CSV and report were saved.', flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    analyze(parser.parse_args().run)
