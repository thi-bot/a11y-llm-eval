import json
from pathlib import Path

import pytest

from scripts.compare_harbor_runs import compare, write_comparison


def reports(tmp_path: Path, outcomes: list[list[str]]) -> list[Path]:
    paths = []
    for i, run in enumerate(outcomes):
        p = tmp_path / f'run-{i}.json'
        p.write_text(json.dumps({
            'run_id': f'run-{i}', 'models': ['codex/model@version [settings]'],
            'results': [{'prompt_case_id': f'case-{j}', 'prompt_variant_id': 'control',
                         'model_name': 'codex/model@version [settings]', 'result': status,
                         'generation': {'prompt_hash': 'same-prompt'}}
                        for j, status in enumerate(run)],
        }))
        paths.append(p)
    return paths


def test_run_median_is_not_prompt_majority(tmp_path):
    paths = reports(tmp_path, [['PASS', 'PASS', 'FAIL'], ['PASS', 'FAIL', 'PASS'], ['FAIL', 'PASS', 'PASS']])
    result = compare(paths)
    row = result['variants'][0]
    assert row['median_run_pass_rate_pct'] == pytest.approx(200 / 3)
    assert row['majority_pass_rate_pct'] == 100
    assert row['mixed_pass_nonpass'] == 3
    assert row['all_three_pass'] == 0
    write_comparison(result, tmp_path / 'comparison')
    assert (tmp_path / 'comparison/cases.csv').is_file()


def test_errors_stay_in_denominator_and_delta_uses_percentage_points(tmp_path):
    paths = reports(tmp_path, [['FAIL', 'ERROR'], ['PASS', 'ERROR'], ['PASS', 'PASS']])
    row = compare(paths)['variants'][0]
    assert row['outputs_per_run'] == 2
    assert row['run_1_errors'] == 1
    assert row['median_run_pass_rate_pct'] == 50
    assert row['median_minus_original_pp'] == 50
    assert row['min_run_pass_rate_pct'] == 0
    assert row['max_run_pass_rate_pct'] == 100


@pytest.mark.parametrize('change', ['coverage', 'duplicate', 'prompt', 'model', 'instructions'])
def test_incompatible_runs_are_rejected(tmp_path, change):
    paths = reports(tmp_path, [['PASS', 'FAIL']] * 3)
    run = json.loads(paths[2].read_text())
    if change == 'coverage':
        run['results'].pop()
    elif change == 'duplicate':
        run['results'].append(run['results'][0])
    elif change == 'prompt':
        run['results'][0]['generation']['prompt_hash'] = 'changed'
    elif change == 'model':
        run['models'] = ['different-model']
    else:
        run['meta'] = {'prompt_variants': [{'custom_instructions': 'Changed instructions'}]}
    paths[2].write_text(json.dumps(run))
    with pytest.raises(ValueError):
        compare(paths)


def test_same_run_cannot_count_three_times(tmp_path):
    paths = reports(tmp_path, [['PASS']] * 3)
    with pytest.raises(ValueError, match='distinct'):
        compare([paths[0]] * 3)
    run = json.loads(paths[1].read_text())
    run['run_id'] = 'run-0'
    paths[1].write_text(json.dumps(run))
    with pytest.raises(ValueError, match='distinct'):
        compare(paths)


def test_skill_turns_stay_separate_and_local_asset_paths_can_differ(tmp_path):
    paths = reports(tmp_path, [['FAIL', 'PASS']] * 3)
    for index, path in enumerate(paths):
        run = json.loads(path.read_text())
        run['meta'] = {'prompt_variants': [{
            'id': 'skill', 'skill_path': f'/run-{index}/skill',
            'custom_instructions_path': f'/run-{index}/instructions.md',
        }]}
        for row, turn in zip(run['results'], ['generate', 'review']):
            row.update(prompt_case_id='same-case', prompt_variant_id='skill', turn_id=turn)
        path.write_text(json.dumps(run))
    rows = compare(paths)['variants']
    assert [(r['turn_id'], r['median_run_pass_rate_pct']) for r in rows] == [
        ('generate', 0), ('review', 100),
    ]


def test_five_run_mean_and_sample_variation(tmp_path):
    paths = reports(tmp_path, [
        ['FAIL', 'FAIL'], ['PASS', 'FAIL'], ['PASS', 'PASS'],
        ['PASS', 'PASS'], ['PASS', 'PASS'],
    ])
    result = compare(paths)
    row = result['variants'][0]
    assert result['n_runs'] == 5
    assert row['mean_run_pass_rate_pct'] == 70
    assert row['median_run_pass_rate_pct'] == 100
    assert row['total_passes'] == 7
    assert row['total_outputs'] == 10
    assert row['sample_variance_pp_squared'] == 2000
    assert row['sample_stddev_pp'] == pytest.approx(2000 ** 0.5)
    assert row['all_runs_pass'] == 0
    assert row['majority_pass_rate_pct'] == 100
    write_comparison(result, tmp_path / 'five')
    assert 'run_5_passes' in (tmp_path / 'five/comparison.csv').read_text()


def test_incomplete_campaign_is_not_published_as_mean(tmp_path):
    paths = reports(tmp_path, [['PASS']] * 5)
    data = json.loads(paths[0].read_text())
    data['meta'] = {'campaign': {'complete': False}}
    paths[0].write_text(json.dumps(data))
    with pytest.raises(ValueError, match='incomplete'):
        compare(paths)
