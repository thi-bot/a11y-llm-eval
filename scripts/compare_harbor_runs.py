#!/usr/bin/env python3
"""Compare independent, single-attempt accessibility benchmark runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev, variance


def _key(row: dict) -> tuple[str, str, str]:
    return (row['prompt_case_id'], row['prompt_variant_id'], row.get('turn_id') or '')


def _metadata(report: dict) -> dict:
    meta = report.get('meta') or {}
    variants = [
        {k: v for k, v in variant.items()
         if k not in {'custom_instructions_path', 'skill_path'}}
        for variant in meta.get('prompt_variants', [])
    ]
    return {'prompts': report.get('prompts'), 'prompt_cases': meta.get('prompt_cases'),
            'prompt_variants': variants, 'models_info': meta.get('models_info'),
            'retry_policy': (meta.get('campaign') or {}).get('policy')}


def compare(paths: list[Path]) -> dict:
    """Return descriptive scores; never treat majority voting as median run rate."""
    paths = [(p / 'results.json' if p.is_dir() else p).resolve() for p in paths]
    count = len(paths)
    if count < 2 or len(set(paths)) != count:
        raise ValueError('Supply at least two distinct results files, original first')
    reports = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    run_ids = [r['run_id'] for r in reports]
    if len(set(run_ids)) != count:
        raise ValueError('Run IDs must be distinct independent runs')
    if len(reports[0]['models']) != 1:
        raise ValueError('Compare one agent/model configuration at a time')
    indexed = []
    for report in reports:
        if (report.get('meta', {}).get('campaign') or {}).get('complete') is False:
            raise ValueError('Campaign run has unresolved infrastructure errors; comparison is incomplete')
        if report['models'] != reports[0]['models'] or _metadata(report) != _metadata(reports[0]):
            raise ValueError('Model configuration or prompt metadata differs between runs')
        rows = {}
        for row in report['results']:
            key = _key(row)
            if key in rows:
                raise ValueError(f'Expected one attempt per prompt/variant/turn: {key}')
            if row['result'] not in {'PASS', 'FAIL', 'ERROR'}:
                raise ValueError(f'Unsupported outcome: {row["result"]}')
            if row['model_name'] != report['models'][0]:
                raise ValueError('Result model does not match report model')
            rows[key] = row
        if not rows:
            raise ValueError('Run has no evaluated outputs')
        if indexed:
            if rows.keys() != indexed[0].keys():
                raise ValueError('Prompt/variant/turn coverage differs between runs')
            for key, row in rows.items():
                original = indexed[0][key]
                if (row.get('prompt_dimensions') != original.get('prompt_dimensions')
                        or row.get('generation', {}).get('prompt_hash')
                        != original.get('generation', {}).get('prompt_hash')):
                    raise ValueError(f'Prompt content or dimensions changed: {key}')
        indexed.append(rows)

    groups = defaultdict(list)
    cases = []
    for key in sorted(indexed[0]):
        outcomes = [rows[key]['result'] for rows in indexed]
        passes = outcomes.count('PASS')
        case = {'prompt_case_id': key[0], 'variant_id': key[1], 'turn_id': key[2],
                **{f'run_{i}': outcome for i, outcome in enumerate(outcomes, 1)},
                f'passes_out_of_{count}': passes, 'passes': passes,
                'majority_pass': passes > count / 2}
        cases.append(case)
        groups[key[1:]].append(case)

    summaries = []
    for (variant, turn), group in sorted(groups.items()):
        n = len(group)
        per_run = [Counter(c[f'run_{i}'] for c in group) for i in range(1, count + 1)]
        rates = [100 * counts['PASS'] / n for counts in per_run]
        middle = median(rates)
        summary = {'variant_id': variant, 'turn_id': turn, 'outputs_per_run': n}
        for i, counts in enumerate(per_run, 1):
            summary.update({f'run_{i}_passes': counts['PASS'],
                            f'run_{i}_errors': counts['ERROR'],
                            f'run_{i}_pass_rate_pct': rates[i - 1]})
        summary.update({
            'mean_run_pass_rate_pct': mean(rates),
            'sample_stddev_pp': stdev(rates),
            'sample_variance_pp_squared': variance(rates),
            'total_passes': sum(c['passes'] for c in group),
            'total_outputs': n * count,
            'median_run_pass_rate_pct': middle,
            'median_minus_original_pp': middle - rates[0],
            'min_run_pass_rate_pct': min(rates), 'max_run_pass_rate_pct': max(rates),
            'all_runs_pass': sum(c['passes'] == count for c in group),
            'all_runs_nonpass': sum(c['passes'] == 0 for c in group),
            'mixed_pass_nonpass': sum(0 < c['passes'] < count for c in group),
            'majority_pass_rate_pct': 100 * sum(c['majority_pass'] for c in group) / n,
        })
        if count == 3:
            summary.update(all_three_pass=summary['all_runs_pass'],
                           all_three_nonpass=summary['all_runs_nonpass'])
        summaries.append(summary)
    return {
        'n_runs': count, 'run_ids': run_ids, 'source_files': [str(p) for p in paths],
        'model': reports[0]['models'][0],
        'method': ('For each variant/turn, arithmetic mean of the full-run PASS rates. '
                   'ERROR stays in the denominator. Majority pass across attempts '
                   'is a separate per-prompt measure. Median and sample standard '
                   'deviation across runs are descriptive secondary statistics.'),
        'variants': summaries, 'cases': cases,
    }


def write_comparison(comparison: dict, out: Path) -> None:
    if out.exists():
        raise ValueError(f'Comparison output already exists: {out}')
    out.mkdir(parents=True)
    (out / 'comparison.json').write_text(
        json.dumps(comparison, indent=2) + '\n', encoding='utf-8')
    for name, rows in [('comparison.csv', comparison['variants']), ('cases.csv', comparison['cases'])]:
        with (out / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path, help='Original report first, then repeats')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.reports)
        write_comparison(result, args.out)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f'Comparison written to {args.out}')


if __name__ == '__main__':
    main()
