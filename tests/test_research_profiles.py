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
