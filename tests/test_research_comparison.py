from copy import deepcopy

import pytest
from PIL import Image

from cfgctrl.benchmark.common import file_digest, read_json, write_json
from cfgctrl.benchmark.config import bind_plan, cells, load_config, samples
from cfgctrl.benchmark.evaluation import METRIC_PROTOCOLS, prepare_jobs
from cfgctrl.benchmark.generation import paths, sample_fingerprint
from scripts.compare_research import check_design, compare, paired_statistics


def test_paired_bootstrap_averages_seeds_within_prompts():
    # Prompt A has two seeds, prompt B one: prompt weights must remain equal.
    stats = paired_statistics([0, 0, 0], [0.5, 1.5, -1], ['a', 'a', 'b'], 1000)
    assert stats['mean_difference'] == 0
    assert stats['prompt_win_rate'] == 0.5
    assert stats['bootstrap_95_ci'] == [-1, 1]
    constant = paired_statistics([0, 1], [0.25, 1.25], ['a', 'b'], 1000)
    assert constant['bootstrap_95_ci'] == [0.25, 0.25]


def make_run(tmp_path, name, arms, score):
    manifest = tmp_path / 'coco.json'
    write_json(manifest, dict(schema_version=1, benchmark='coco', records=[
        dict(id=f'p{i}', image_id=i, prompt=f'caption {i}', reference=f'{i}.png') for i in range(2)]))
    config = tmp_path / f'{name}.json'
    write_json(config, dict(schema_version=1, models={'sd35': {'width': 16, 'height': 16}},
                           arms=arms, benchmarks={'coco': dict(manifest='coco.json',
                                                              seeds=[0, 1], metrics=['clip'])}))
    plan, base = load_config(config)
    run = tmp_path / name
    fingerprint = bind_plan(run, plan, base)
    write_json(run / 'generation_environment.json', {'torch': 'test'})
    for cell in cells(plan):
        for item in samples(plan, cell):
            image, record = paths(run, cell, item)
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (16, 16)).save(image)
            write_json(record, dict(fingerprint=sample_fingerprint(fingerprint, cell, item),
                                    image_sha256=file_digest(image)))
    jobs = prepare_jobs(run, plan, fingerprint, base, metrics=['clip'])
    for task in jobs['clip']:
        write_json(task['output'], dict(metric='clip', protocol=METRIC_PROTOCOLS['clip'],
            input_fingerprint=task['input_fingerprint'], resource=task['resource'], versions={},
            scores=[dict(id=r['id'], score=score) for r in task['records']]))
    return run, plan


def test_saved_runs_compare_without_inference_and_reject_stale_scores(tmp_path):
    reference, ref_plan = make_run(tmp_path, 'reference', ['cfg', 'paper'], 0.25)
    candidate, new_plan = make_run(tmp_path, 'candidate', ['rel03=proximal_relative:k=0.3'], 0.375)
    report = compare(reference, candidate, ['cfg', 'paper'], ['clip'], 'sd35', 1000)
    assert len(report['results']) == 2
    assert all(r['mean_difference'] == 0.125 for r in report['results'])
    metric = candidate / 'sd35/coco/rel03/w7.5/metrics/clip.json'
    data = read_json(metric)
    data['input_fingerprint'] = 'outdated'
    write_json(metric, data)
    with pytest.raises(ValueError, match='Stale metric'):
        compare(reference, candidate, ['cfg'], ['clip'], 'sd35', 1000)
    for key in ('models', 'generation_implementation', 'benchmarks'):
        changed = deepcopy(new_plan)
        if key == 'models':
            changed[key]['sd35']['steps'] += 1
        elif key == 'generation_implementation':
            changed[key]['controllers.py'] = 'changed'
        else:
            changed[key]['coco']['manifest']['records'][0]['prompt'] = 'different caption'
        with pytest.raises(ValueError):
            check_design(ref_plan, changed, 'sd35')
