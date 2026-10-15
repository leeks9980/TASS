"""Validate recorded visual signal propagation and create a readable HTML/SVG report.

This is an observational validator.  It never advances the CNS model and it does
not claim that population means prove a causal serial pathway.
"""
import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from full_cns_recording import state_at
from neurotransmitter_model import neurotransmitter_effects


STAGES = [
    ('photoreceptor', '광수용체', 'ol_sensory'),
    ('optic_lobe', '시각엽 고유 뉴런', 'ol_intrinsic'),
    ('visual_projection', '시각 투사 뉴런', 'visual_projection'),
    ('descending', '하행 뉴런 (DN)', 'descending_neuron'),
    ('leg_motor', '다리 운동뉴런', 'vnc_motor'),
]
COLORS = ['#38bdf8', '#a78bfa', '#fb7185', '#fbbf24', '#34d399']


def load_events(run):
    rows = []
    per_case = {}
    for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        case = event['case']
        event['frame_in_case'] = per_case.get(case, 0)
        per_case[case] = event['frame_in_case'] + 1
        rows.append(event)
    return rows


def safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def extract(run, annotation, events):
    side = annotation.rootSide.fillna(annotation.somaSide).fillna('').astype(str)
    motor = annotation.superclass.eq('vnc_motor') & annotation.subclass.isin(['fl', 'ml', 'hl'])
    masks = []
    counts = {}
    for key, label, superclass in STAGES:
        base = motor if key == 'leg_motor' else annotation.superclass.eq(superclass)
        counts[key] = int(base.sum())
        for suffix, side_value in [('all', None), ('L', 'L'), ('R', 'R')]:
            mask = base if side_value is None else base & side.eq(side_value)
            masks.append((key, label, suffix, mask.to_numpy()))
    selected_mask = np.logical_or.reduce([item[3] for item in masks])
    selected_positions = np.flatnonzero(selected_mask)
    graph_indices = annotation.graph_index.to_numpy(dtype=np.int64)[selected_positions]
    local_masks = [(key, label, suffix, mask[selected_positions]) for key, label, suffix, mask in masks]
    rows = []
    for number, event in enumerate(events, 1):
        with np.load(run / event['snapshot']) as snapshot:
            values = state_at(snapshot, graph_indices).astype(np.float64)
            input_indices = snapshot['input_indices'].astype(np.int64)
            input_values = snapshot['input_values'].astype(np.float64)
        for key, label, suffix, mask in local_masks:
            subset = values[mask]
            rows.append({
                'case': event['case'], 'frame': event['frame'],
                'frame_in_case': event['frame_in_case'],
                'time_s': float(event.get('input_sim_seconds', event['sim_seconds'])),
                'stage': key, 'stage_label': label, 'side': suffix,
                'neurons': int(len(subset)), 'mean': safe_mean(subset),
                'max': float(np.max(subset)) if len(subset) else 0.0,
                'active_fraction': float(np.count_nonzero(subset) / len(subset)) if len(subset) else 0.0,
                'saturated_fraction': float(np.count_nonzero(subset >= .999) / len(subset)) if len(subset) else 0.0,
                'finite': bool(np.isfinite(subset).all()),
            })
        # Direct camera drive is stored separately from the recurrent state.
        input_lookup = pd.Series(input_values, index=input_indices)
        photo = annotation[annotation.superclass.eq('ol_sensory')][['graph_index']].copy()
        photo['side'] = side[annotation.superclass.eq('ol_sensory')].to_numpy()
        driven = photo.graph_index.map(input_lookup).fillna(0.).to_numpy(dtype=float)
        for suffix, keep in [('all', np.ones(len(photo), dtype=bool)),
                             ('L', photo.side.eq('L').to_numpy()),
                             ('R', photo.side.eq('R').to_numpy())]:
            subset = driven[keep]
            rows.append({'case': event['case'], 'frame': event['frame'],
                         'frame_in_case': event['frame_in_case'],
                         'time_s': float(event.get('input_sim_seconds', event['sim_seconds'])),
                         'stage': 'camera_drive', 'stage_label': '카메라→광수용체 입력', 'side': suffix,
                         'neurons': int(len(subset)), 'mean': safe_mean(subset),
                         'max': float(np.max(subset)) if len(subset) else 0.,
                         'active_fraction': float(np.count_nonzero(subset) / len(subset)) if len(subset) else 0.,
                         'saturated_fraction': float(np.count_nonzero(subset >= .999) / len(subset)) if len(subset) else 0.,
                         'finite': bool(np.isfinite(subset).all())})
        print(f'신호 단계 검증 {number}/{len(events)}: {event["case"]} {event["frame_in_case"]}', flush=True)
    return pd.DataFrame(rows), counts


def add_neutral_delta(table):
    neutral = table[table.case.eq('neutral')][['frame_in_case', 'stage', 'side', 'mean']].rename(
        columns={'mean': 'neutral_mean'})
    result = table.merge(neutral, on=['frame_in_case', 'stage', 'side'], how='left', validate='many_to_one')
    fallback = table[table.case.eq('neutral')].groupby(['stage', 'side'])['mean'].mean()
    missing = result.neutral_mean.isna()
    if missing.any():
        result.loc[missing, 'neutral_mean'] = [fallback.get((s, d), 0.) for s, d in
                                               zip(result.loc[missing, 'stage'], result.loc[missing, 'side'])]
    result['signed_delta'] = result['mean'] - result['neutral_mean']
    result['absolute_delta'] = result.signed_delta.abs()
    return result


def svg_heatmap(summary, stages, cases):
    width, cell, left, top = 1000, 105, 190, 70
    height = top + 58 * len(cases) + 70
    max_value = max(float(summary.absolute_delta.max()), 1e-12)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
             '<style>text{font-family:Segoe UI,Malgun Gothic,sans-serif;fill:#dbeafe}.small{font-size:13px}.label{font-size:15px}</style>',
             f'<rect width="{width}" height="{height}" rx="18" fill="#0f172a"/>',
             '<text x="28" y="35" font-size="21" font-weight="700">Neutral 대비 단계별 평균 변화량</text>']
    for column, stage in enumerate(stages):
        label = dict((k, l) for k, l, _ in STAGES).get(stage, '입력')
        parts.append(f'<text class="small" text-anchor="middle" x="{left+column*cell+cell/2}" y="58">{html.escape(label)}</text>')
    for row, case in enumerate(cases):
        y = top + row * 58
        parts.append(f'<text class="label" x="28" y="{y+32}">{html.escape(case)}</text>')
        for column, stage in enumerate(stages):
            match = summary[(summary.case == case) & (summary.stage == stage)]
            value = float(match.absolute_delta.iloc[0]) if len(match) else 0.
            ratio = min(1., value / max_value)
            red, green, blue = int(30+210*ratio), int(64+55*(1-ratio)), int(175-105*ratio)
            x = left + column * cell
            parts.append(f'<rect x="{x+3}" y="{y+3}" width="{cell-6}" height="46" rx="7" fill="rgb({red},{green},{blue})"/>')
            parts.append(f'<text class="small" text-anchor="middle" x="{x+cell/2}" y="{y+31}">{value:.3g}</text>')
    parts.append(f'<text class="small" x="28" y="{height-22}">값: 자극 시작 이후 |조건 평균 − neutral 평균|. 크기 자체는 생물학적 단위를 뜻하지 않습니다.</text></svg>')
    return ''.join(parts)


def svg_side_bars(summary, stages):
    cases = [case for case in ['move_left', 'move_right'] if case in set(summary.case)]
    width, left = 1000, 210
    height = 110 + len(cases) * (45 * len(stages) + 24)
    max_value = max(float(summary.absolute_delta.max()), 1e-12)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Segoe UI,Malgun Gothic,sans-serif;fill:#dbeafe}.s{font-size:13px}</style>',
             f'<rect width="{width}" height="{height}" rx="18" fill="#0f172a"/>',
             '<text x="28" y="36" font-size="21" font-weight="700">좌·우 움직임 조건의 좌/우 집단 반응</text>']
    y = 75
    for case in cases:
        parts.append(f'<text x="28" y="{y+18}" font-size="16" font-weight="700">{case}</text>')
        for stage_index, stage in enumerate(stages):
            label = dict((k, l) for k, l, _ in STAGES).get(stage, stage)
            parts.append(f'<text class="s" x="{left-10}" y="{y+18}" text-anchor="end">{html.escape(label)}</text>')
            for side, color, dy in [('L', '#38bdf8', 2), ('R', '#fb7185', 22)]:
                match = summary[(summary.case == case) & (summary.stage == stage) & (summary.side == side)]
                value = float(match.absolute_delta.iloc[0]) if len(match) else 0.
                bar = 650 * value / max_value
                parts.append(f'<rect x="{left}" y="{y+dy}" width="{bar:.1f}" height="15" rx="4" fill="{color}"/>')
                parts.append(f'<text class="s" x="{left+bar+7:.1f}" y="{y+dy+12}">{side} {value:.3g}</text>')
            y += 45
        y += 24
    parts.append(f'<text class="s" x="28" y="{height-20}" fill="#94a3b8">파랑=L 집단, 분홍=R 집단. 방향 선택성의 존재 여부를 눈으로 비교하기 위한 기술 통계입니다.</text></svg>')
    return ''.join(parts)


def validate(run):
    run = run.resolve()
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    events = load_events(run)
    if not events or 'neutral' not in {e['case'] for e in events}:
        raise ValueError('neutral을 포함한 기록이 필요합니다.')
    annotation = feather.read_feather(run / 'annotations.feather')
    timeline, counts = extract(run, annotation, events)
    timeline = add_neutral_delta(timeline)
    onset = float(metadata.get('parameters', {}).get('stimulus_onset', .5))
    post = timeline[(timeline.time_s >= onset) & timeline.side.eq('all') & ~timeline.case.eq('neutral')]
    summary = post.groupby(['case', 'stage', 'stage_label'], as_index=False).agg(
        mean=('mean', 'mean'), neutral_mean=('neutral_mean', 'mean'),
        signed_delta=('signed_delta', 'mean'), absolute_delta=('absolute_delta', 'mean'),
        peak_absolute_delta=('absolute_delta', 'max'), active_fraction=('active_fraction', 'mean'),
        saturated_fraction=('saturated_fraction', 'mean'))
    side_summary = timeline[(timeline.time_s >= onset) & ~timeline.side.eq('all') & ~timeline.case.eq('neutral')].groupby(
        ['case', 'stage', 'stage_label', 'side'], as_index=False).agg(
        signed_delta=('signed_delta', 'mean'), absolute_delta=('absolute_delta', 'mean'))
    output = run / 'signal_chain_validation'
    output.mkdir(exist_ok=True)
    timeline.to_csv(output / 'stage_timeseries.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(output / 'condition_summary.csv', index=False, encoding='utf-8-sig')
    side_summary.to_csv(output / 'side_summary.csv', index=False, encoding='utf-8-sig')

    current = neurotransmitter_effects('negative')
    recorded = metadata.get('nt', {}).get('modeled_effects', {})
    required = {k: current[k] for k in ['acetylcholine', 'gaba', 'glutamate', 'histamine']}
    profile_matches = all(float(recorded.get(k, 999.)) == float(v) for k, v in required.items())
    finite = bool(timeline.finite.all())
    stage_signal = {stage: float(summary[summary.stage.eq(stage)].absolute_delta.max())
                    if (summary.stage == stage).any() else 0. for stage, _, _ in STAGES}
    warnings = []
    if not profile_matches:
        warnings.append('이 기록은 현재 전달물질 부호 설정과 다릅니다. 코드 검사용으로만 보고, 최종 검증은 재수집 기록으로 해야 합니다.')
    if not finite:
        warnings.append('NaN 또는 무한대 상태가 발견되었습니다.')
    for stage, value in stage_signal.items():
        if value <= 1e-12:
            warnings.append(f'{stage} 단계에서 neutral 대비 측정 가능한 변화가 없습니다.')
    verdict = '재수집 필요' if not profile_matches else ('수치 오류 발견' if not finite else '검토 가능')
    report = {
        'schema': 1, 'run': str(run), 'verdict': verdict,
        'recording_matches_current_transmitter_profile': profile_matches,
        'recorded_effects': recorded, 'current_required_effects': required,
        'equation': metadata.get('equation'), 'stimulus_onset_s': onset,
        'frames': len(events), 'conditions': sorted({e['case'] for e in events}),
        'population_counts': counts, 'maximum_post_onset_absolute_delta_by_stage': stage_signal,
        'all_values_finite': finite, 'warnings': warnings,
        'interpretation_limits': [
            '집단 평균은 신호가 해당 집단에 나타났는지 보는 기술 통계이며 개별 시냅스의 인과 경로 증명이 아닙니다.',
            'DN과 다리 운동뉴런의 반응은 실제 보행 또는 근육 구동 검증이 아닙니다.',
            '좌우 차이와 자극 선택성은 공개 생리 데이터와 별도로 비교해야 합니다.',
            '현재 비음수·0 기준선 모델에서는 억제성 히스타민 입력이 잘릴 수 있으므로 새 기록에서 광수용체 다음 단계 소실 여부를 반드시 확인해야 합니다.'
        ]}
    (output / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    ordered_stages = ['camera_drive'] + [key for key, _, _ in STAGES]
    cases = [c for c in ['bright', 'dim', 'move_left', 'move_right', 'approach', 'free'] if c in set(summary.case)]
    heatmap = svg_heatmap(summary, ordered_stages, cases)
    sides = svg_side_bars(side_summary, [key for key, _, _ in STAGES])
    (output / 'signal_chain_heatmap.svg').write_text(heatmap, encoding='utf-8')
    (output / 'left_right_response.svg').write_text(sides, encoding='utf-8')
    warning_html = ''.join(f'<li>{html.escape(x)}</li>' for x in warnings) or '<li>자동 경고 없음</li>'
    rows_html = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>'
                        for row in summary[['case', 'stage_label', 'signed_delta', 'absolute_delta', 'active_fraction']].itertuples(index=False, name=None))
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>시각 신호 검증 보고서</title>
<style>body{{margin:auto;max-width:1120px;padding:36px;background:#020617;color:#e2e8f0;font-family:Segoe UI,Malgun Gothic,sans-serif}}h1,h2{{color:#f8fafc}}.status{{display:inline-block;padding:8px 14px;border-radius:99px;background:#7c2d12;font-weight:700}}.card{{background:#0f172a;border:1px solid #334155;border-radius:16px;padding:20px;margin:18px 0}}img{{width:100%}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #334155;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}code{{color:#7dd3fc}}</style>
<h1>시각 입력 → 다리 운동뉴런 신호 검증</h1><p class="status">판정: {html.escape(verdict)}</p>
<div class="card"><h2>무엇을 검사했나</h2><p>몸체는 고정된 기록입니다. 카메라가 만든 광수용체 입력과 CNS 다섯 집단의 값을 neutral 조건과 같은 프레임 위치에서 비교했습니다. 실제 움직임이나 디코더는 검증 대상이 아닙니다.</p><ul>{warning_html}</ul></div>
<div class="card">{heatmap}</div><div class="card">{sides}</div>
<div class="card"><h2>수치 표</h2><table><thead><tr><th>조건</th><th>단계</th><th>부호 변화</th><th>절대 변화</th><th>활성 비율</th></tr></thead><tbody>{rows_html}</tbody></table></div>
<div class="card"><h2>파일 안내</h2><p><code>stage_timeseries.csv</code>: 모든 프레임·단계·좌우 값<br><code>condition_summary.csv</code>: 조건별 요약<br><code>side_summary.csv</code>: 좌우 비교<br><code>validation_report.json</code>: 자동 판정과 한계</p></div></html>'''
    (output / 'report.html').write_text(document, encoding='utf-8')
    print(f'검증 보고서 저장: {output / "report.html"}', flush=True)
    return report


def latest_run():
    candidates = [p for p in (ROOT / 'simulation_logs').glob('full_fly_*')
                  if (p / 'events.jsonl').exists() and (p / 'metadata.json').exists()]
    if not candidates:
        raise FileNotFoundError('검증할 full_fly 기록이 없습니다.')
    return max(candidates, key=lambda p: p.stat().st_mtime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', type=Path, help='생략하면 가장 최근 full_fly 기록')
    args = parser.parse_args()
    validate(args.run or latest_run())
