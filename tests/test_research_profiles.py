import json

import pytest

from scripts.prepare_research import prepare


def source_config(tmp_path):
    manifest = tmp_path / 'coco.json'
    manifest.write_text(json.dumps(dict(schema_version=1, benchmark='coco', records=[
        dict(id=f'coco_{i}', image_id=i, reference=f'{i}.jpg', prompt=f'Original caption {i}')
        for i in range(400)])))
    source = tmp_path / 'paper.json'
    source.write_text(json.dumps(dict(schema_version=1, models={'sd35': {}},
        benchmarks={'coco': {'manifest': 'coco.json'}},
        evaluation={'python': '../metrics/bin/python',
                    'scorers': {'compbench': {'repo': 'unused'}}})))
    return source


def test_research_splits_preserve_pairs_and_are_disjoint(tmp_path):
    source = source_config(tmp_path)
    outputs = {}
    for stage in ('smoke', 'dev', 'validation'):
        output = tmp_path / 'profiles' / f'{stage}.json'
        config = prepare(source, output, stage)
        rows = json.loads(output.with_suffix('.manifest.json').read_text())['records']
        outputs[stage] = {r['id'] for r in rows}
        assert all(r['prompt'] == f'Original caption {r["image_id"]}' for r in rows)
        assert config['evaluation']['python'] == str((tmp_path / '../metrics/bin/python').resolve())
        assert config['evaluation']['scorers'] == {}
        assert config['benchmarks']['coco']['metrics'] == ['clip', 'pickscore']
    assert len(outputs['dev']) == 64
    assert len(outputs['validation']) == 256
    assert outputs['smoke'] <= outputs['dev']
    assert outputs['dev'].isdisjoint(outputs['validation'])


def test_research_rerun_preserves_config_and_refuses_changed_arms(tmp_path):
    source = source_config(tmp_path)
    output = tmp_path / 'dev.json'
    prepare(source, output, 'dev')
    before = output.read_bytes()
    prepare(source, output, 'dev')
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match='different settings'):
        prepare(source, output, 'dev', ['cfg', 'paper'])
    assert output.read_bytes() == before


def test_single_arm_copies_frozen_run_instead_of_reselecting(tmp_path):
    from cfgctrl.benchmark.config import bind_plan, cells, load_config
    source = source_config(tmp_path)
    dev = tmp_path / 'dev.json'
    prepare(source, dev, 'dev')
    original, base = load_config(dev)
    run = tmp_path / 'old_run'
    bind_plan(run, original, base)
    # Source configuration changes must not affect a saved-run variant.
    source.unlink()
    variant = tmp_path / 'rel03.json'
    prepare(source, variant, 'dev', ['rel03=proximal_relative:k=0.3'], from_run=run)
    plan, _ = load_config(variant)
    assert plan['models'] == original['models']
    assert plan['benchmarks']['coco']['manifest'] == original['benchmarks']['coco']['manifest']
    assert plan['benchmarks']['coco']['seeds'] == original['benchmarks']['coco']['seeds']
    selected = list(cells(plan))
    assert len(selected) == 1
    assert selected[0]['controller']['k'] == 0.3
    assert selected[0]['controller']['relative_gain'] is True


def test_research_model_selection(tmp_path):
    source = source_config(tmp_path)
    raw = json.loads(source.read_text())
    raw['models']['flux'] = {'scales': [2.0], 'steps': 20}
    source.write_text(json.dumps(raw))
    result = prepare(source, tmp_path / 'flux.json', 'smoke', ['cfg'], model='flux')
    assert list(result['models']) == ['flux']
    assert result['models']['flux']['steps'] == 20
