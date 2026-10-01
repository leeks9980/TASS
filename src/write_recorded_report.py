"""Summarize existing offline diagnosis outputs; never run the simulator."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd


def main():
    folders = sorted((ROOT / 'simulation_logs').glob('full_cns_*'))
    summaries = []
    for folder in folders:
        summary = folder / 'diagnosis' / 'summary.json'
        if summary.exists():
            value = json.loads(summary.read_text(encoding='utf-8'))
            if value.get('output_mode', 'wheel_mean_proxy') == 'wheel_mean_proxy':
                summaries.append((folder, value))
    if not summaries:
        raise ValueError('이 보고서 작성기는 기존 바퀴 모델 기록용입니다. 근육 모델 기록은 inspect_recorded_run.py로 분석하세요.')
    folder, s = max(summaries, key=lambda pair: pair[1]['last_sim_seconds'])
    out = folder / 'diagnosis'
    timeline = pd.read_csv(out / 'timeline.csv')
    trace = pd.read_csv(folder / 'analysis' / 'trace_frame_32_body_804847.csv')
    dn_trace = pd.read_csv(folder / 'analysis' / 'trace_frame_31_body_10264.csv')
    trace['pre_superclass'] = trace.pre_superclass.fillna('unannotated')
    trace['positive'] = trace.modeled_signal_contribution.clip(lower=0)
    trace['negative'] = -trace.modeled_signal_contribution.clip(upper=0)
    by_group = trace.groupby('pre_superclass')[['positive', 'negative', 'modeled_signal_contribution']].sum()
    by_group.to_csv(out / 'strongest_motor_source_groups.csv', encoding='utf-8-sig')
    dn_trace['pre_superclass'] = dn_trace.pre_superclass.fillna('unannotated')
    dn_groups = dn_trace.groupby('pre_superclass').modeled_signal_contribution.sum().sort_values(ascending=False)
    dn_groups.to_csv(out / 'DNp31_source_groups.csv', encoding='utf-8-sig')
    audits = []
    for run, summary in summaries:
        rows = pd.read_csv(run / 'diagnosis' / 'timeline.csv')
        score_error = max(max(abs(ast.literal_eval(row.motor_scores)[side] - row[f'{name}_motor_mean_recomputed'])
                              for side, name in [('L', 'left'), ('R', 'right')])
                          for _, row in rows.iterrows())
        audits.append(dict(run=run.name, frames=summary['frames_analyzed'],
                           sim_seconds=summary['last_sim_seconds'], path_mm=summary['final_path_m']*1000,
                           motor_score_record_error=score_error,
                           motor_update_reconstruction_error=summary['mn_reconstruction_max_error']))
    pd.DataFrame(audits).to_csv(out / 'runs_compared.csv', index=False, encoding='utf-8-sig')
    commands = np.array(s['final_motor_commands'])
    velocities = np.array(s['final_wheel_velocity'])
    ratios = velocities/commands
    summed_seconds = float((timeline.compute_ms_excluding_record + timeline.save_ms).sum()/1000)
    negatives = s['mn_negative_drive_mean']/s['mn_positive_drive_mean']
    lines = [
        '# 기존 바퀴 모델 기록 분석 결과',
        '',
        '새 시뮬레이션, 학습, 신경망 상태 진행 없이 저장된 NPZ·이벤트·실행 설정과 원본 연결표를 조회했습니다. 시뮬레이션 코드는 변경하지 않았습니다.',
        '',
        f'분석 대상은 기존 바퀴 출력 모델의 전체 CNS 실행 기록 {len(summaries)}개, 저장 프레임 총 {sum(x["frames_analyzed"] for _, x in summaries)}개입니다. 새 근육 출력 모델은 이 보고서에 포함하지 않습니다. 모두 `free` 조건, 에피소드 0입니다. 비교 대상은 별도 실행이며 시간을 이어 붙이지 않았습니다.',
        '',
        '| 기록 | 프레임 | 마지막 시뮬레이션 시간 | 누적 이동 거리 |',
        '|---|---:|---:|---:|',
    ]
    for row in audits:
        lines.append(f'| {row["run"]} | {row["frames"]} | {row["sim_seconds"]:.2f} s | {row["path_mm"]:.9g} mm |')
    lines += [
        '',
        f'이하 상세 수치는 가장 긴 기록 `{folder.name}`의 마지막 프레임 32, 시뮬레이션 시간 1.65초 기준입니다. 최신 실행은 1.05초까지만 기록되어 있어 가장 긴 기록을 상세 분석 대상으로 삼았습니다.',
        '',
        '## 확인된 신경 활동',
        '',
        '활동값은 현재 모델의 무차원 수치이며 실제 발화율이나 막전압으로 보정되어 있지 않습니다. 활성 개수는 단순히 0보다 큰 값의 개수이며 생물학적 반응 임계치를 뜻하지 않습니다.',
        '',
        '| 주석 집단 | 개수 | 0보다 큰 활동 | 평균 활동 | 최대 활동 |',
        '|---|---:|---:|---:|---:|',
    ]
    labels = {'L1': 'L1', 'L2': 'L2', 'L3': 'L3', 'visual_projection': '시각 투사신경',
              'descending_neuron': '하행신경', 'vnc_intrinsic': 'VNC 내부신경', 'leg_motor': '다리 운동신경'}
    for group, label in labels.items():
        values = s['final_group_stats'][group]
        lines.append(f'| {label} | {values["n"]:,} | {values["active"]:,} | {values["mean"]:.6g} | {values["max"]:.6g} |')
    lines += [
        '',
        '시각 입력과 하행신경, 다리 운동신경에 기록된 활동이 있습니다. 다리 운동신경에서도 전체 381개 중 370개가 0보다 크므로 출력 연결이 완전히 끊긴 상태는 아닙니다. 집단 평균끼리의 비교는 단일한 직렬 경로의 감쇠율을 입증하지 않습니다.',
        '',
        '## 움직임이 안 보이는 직접적인 이유',
        '',
        '| 항목 | 왼쪽 | 오른쪽 |',
        '|---|---:|---:|',
        f'| 운동신경 평균 | {s["final_group_stats"]["leg_motor_L"]["mean"]:.9g} | {s["final_group_stats"]["leg_motor_R"]["mean"]:.9g} |',
        f'| 모터 명령(rad/s) | {commands[0]:.9g} | {commands[1]:.9g} |',
        f'| 실제 바퀴 속도(rad/s) | {velocities[0]:.9g} | {velocities[1]:.9g} |',
        f'| 명령 대비 실제 속도 | {ratios[0]*100:.2f}% | {ratios[1]*100:.2f}% |',
        '',
        f'모터 명령은 코드의 `각 측 다리 운동신경 평균 × 12`와 일치합니다. 최종 몸체 전진 속도는 {s["final_body_x_velocity"]*1000:.6f} mm/s, 누적 이동 거리는 {s["final_path_m"]*1000:.6f} mm입니다. 바퀴는 명령의 약 95% 수준으로 회전하므로 기록상 물리 모델이 명령을 무시하고 있는 것은 아닙니다. 명령 자체가 육안으로 확인하기 어려울 만큼 작습니다.',
        '',
        '모든 다리 운동신경을 측별 평균으로 합치는 현재 방식은 근육별·굴곡/신전별·보행 위상별 제어를 구분하지 않습니다. 이 값은 아직 검증된 두 모터 행동 명령이 아닙니다. 신호를 단순 증폭해도 장애물 회피 능력이 증명되지는 않습니다.',
        '',
        '## 운동신경 입력의 역추적',
        '',
        '직전 프레임의 실제 상태, 원본 시냅스 수, 기록된 입력 정규화와 전달물질 부호를 이용해 381개 운동신경의 마지막 업데이트를 재구성했습니다. 신경망을 새로 진행한 결과가 아니라 저장된 한 단계의 입력 계산입니다.',
        '',
        f'- 기록값과 재구성값의 최대 절대 오차: {s["mn_reconstruction_max_error"]:.6g}.',
        f'- 양의 입력 기여 평균: {s["mn_positive_drive_mean"]:.9g}.',
        f'- 음의 입력 기여 크기 평균: {s["mn_negative_drive_mean"]:.9g}.',
        f'- 순입력 평균: {s["mn_net_drive_mean"]:.9g}.',
        f'- 집단 평균 기준 음의 기여는 양의 기여의 {negatives*100:.2f}%. 순입력이 0 이하인 운동신경은 {s["mn_net_drive_nonpositive"]}/381개입니다.',
        '',
        '따라서 운동신경 전체가 억제로 차단된 결과라고 해석할 근거도 없습니다. 현재 모델은 입력 시냅스 수로 정규화한 가중 평균에 gain=0.95와 leak=0.65를 적용합니다. 입력 신호의 크기는 이 모델 가정에 의존하며 원본 시냅스 수 자체가 생리학적 전달 강도로 보정된 것은 아닙니다.',
        '',
        '가장 활동이 큰 다리 운동신경은 `bodyId=804847`, 주석 `MNhl88`, 왼쪽 뒷다리, 마지막 활동 0.000234449입니다. 이 신경의 연결별 양·음 기여 상위 항목은 다음과 같습니다. 방향이나 회피 기능의 입증이 아니라 현재 모델 계산 안에서의 기여입니다.',
        '',
        '| 입력 body ID | 주석 | 시냅스 수 | 기록된 직전 활동 | 모델 입력 기여 |',
        '|---|---|---:|---:|---:|',
    ]
    for _, row in trace.head(8).iterrows():
        label = row.pre_type if pd.notna(row.pre_type) else '미주석'
        lines.append(f'| {int(row.body_pre)} | {label} | {int(row.synapse_count)} | {row.previous_activity:.6g} | {row.modeled_signal_contribution:+.6g} |')
    lines += [
        '',
        '상위 양의 기여에 하행신경 DNp31·DNg03·DNp03이 포함됩니다. DNp31(bodyId 10264)의 프레임 31 입력도 한 단계 더 역추적해 CSV로 저장했습니다.',
        '',
        '실제로 확인한 연결 예시는 `vCal3(bodyId 10955) → DNp31(bodyId 10264) → MNhl88(bodyId 804847)`입니다. 첫 연결은 시냅스 196개, 직전 활동 0.0189514, 모델 입력 기여 +0.000192104입니다. 두 번째 연결은 시냅스 14개, 직전 활동 0.00500413, 모델 입력 기여 +0.000025998입니다. 연속한 두 업데이트의 저장 상태를 기준으로 확인했습니다.',
        '',
        f'DNp31의 입력을 집단별로 합친 시각 투사신경의 순기여는 {dn_groups.get("visual_projection", 0.):.9g}입니다. 현재 모델에서 시각 관련 집단의 활동이 하행신경을 통해 운동신경 입력에 기여한 구체적 예시를 확보한 것입니다. 장애물 회피를 담당한다는 생물학적 기능을 입증한 것은 아닙니다. 이들은 후속 조건 비교를 위한 관찰 후보이며 현재 기록만으로 필요한 회로로 확정할 수 없습니다.',
        '', '## 기록 시간과 해석 범위', '',
    ]
    lines += [
        f'가장 긴 실행의 프레임당 계산 시간 중앙값은 {s["median_compute_ms"]/1000:.3f}초, 저장 시간 중앙값은 {s["median_save_ms"]/1000:.3f}초입니다. 기록된 계산+저장 시간 합은 {summed_seconds:.2f}초지만 진행된 가상 시간은 1.65초뿐입니다. 이 합은 일시정지·뷰어 대기 시간을 포함한 전체 실제 실행 시간이 아닙니다. 현재 전체 그래프 기록 방식은 실시간 속도로 작동하지 않습니다.',
        '',
        '모든 실행에 무물체 기준 조건, 별도 밝기 변화 조건, 좌우 이동 비교 조건이 없습니다. 신경 상태는 0에서 시작했고 마지막까지 운동신경 활동이 증가하고 있습니다. 초기 상태가 채워지는 과정과 물체 이동에 의한 반응을 이 기록만으로 분리할 수 없습니다.',
        '',
        '현재 결과로 말할 수 있는 것은 “시각 입력에서 다리 운동신경까지 모델 활동이 전달되며, 미세한 실제 움직임이 기록되었다”입니다. 물체를 장애물로 인식했는지, 회피를 결정했는지, 목적성 부족 때문에 멈췄는지는 이 기록으로 판정할 수 없습니다.',
        '',
        '미주석 graph ID는 완전한 뉴런으로 확인되지 않은 segment를 포함합니다. 전달물질 예측은 모델의 부호 선택에 이용되지만 기타/불명 전달물질은 현재 +1로 가정되며, 도파민의 동기·보상·학습 작용을 재현하는 모델은 없습니다.',
        '',
        '## 결과 파일',
        '',
        '- `runs_compared.csv`: 4개 실행의 시간·거리 및 재구성 오차.',
        '- `timeline.csv`: 가장 긴 실행의 전체 33프레임 입력·출력·시간.',
        '- `population_activity.csv`: 프레임별 주석 신경 집단 통계.',
        '- `motor_neurons.csv`: 운동신경 381개의 개별 활동과 ID.',
        '- `motor_input_balance.csv`: 운동신경별 양·음 입력과 재구성값.',
        '- `strongest_motor_source_groups.csv`, `DNp31_source_groups.csv`: 추적한 신경의 입력 집단별 기여.',
        '- 각 실행의 `diagnosis/`에 해당 실행의 분석 결과가 있습니다.',
        '- 한 단계 연결별 상세 추적은 가장 긴 실행의 `analysis/trace_frame_32_body_804847.csv` 및 `analysis/trace_frame_31_body_10264.csv`입니다.',
        '',
    ]
    report = out / 'recorded_analysis_ko.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(report)
    print(json.dumps({'runs': audits, 'wheel_command_ratios': ratios.tolist(),
                     'recorded_compute_and_save_seconds': summed_seconds,
                     'strongest_motor_source_groups': by_group.to_dict(),
                     'DNp31_source_groups': dn_groups.to_dict()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
