"""Create small, reproducible controller experiments from an existing configuration."""
import argparse
import copy
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfgctrl.benchmark.common import file_digest, read_json, resolve, write_json
from cfgctrl.benchmark.config import load_config
from cfgctrl.benchmark.datasets import save_manifest, validate_manifest

STAGES = {'smoke': (0, 8, [0]), 'dev': (0, 64, [0]),
          'validation': (64, 256, [0, 1])}


def prepare(source, output, stage, arms=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('Research configuration must have a separate output path')
    original = read_json(source)
    manifest_path = resolve(source.parent, original['benchmarks']['coco']['manifest'])
    manifest = validate_manifest(read_json(manifest_path))
    if manifest['benchmark'] != 'coco':
        raise ValueError('Research profiles require a COCO source manifest')
    start, count, seeds = STAGES[stage]
    records = sorted(manifest['records'], key=lambda r: r['id'])
    if len(records) < start + count:
        raise ValueError(f'{stage} requires at least {start + count} source prompts')
    random.Random(0).shuffle(records)
    selected = sorted(records[start:start + count], key=lambda r: r['id'])
    subset_path = output.with_suffix('.manifest.json')
    config = dict(schema_version=1,
                  models={'sd35': copy.deepcopy(original['models']['sd35'])},
                  arms=arms or ['cfg', 'paper', 'proximal', 'proximal_relative:k=0.1'],
                  benchmarks={'coco': dict(manifest=subset_path.name, seeds=seeds,
                                            metrics=['clip', 'pickscore'])},
                  evaluation=copy.deepcopy(original.get('evaluation', {})))
    # Retain the user's installed evaluator interpreters when moving the config.
    evaluation = config['evaluation']
    evaluation['scorers'] = {k: v for k, v in evaluation.get('scorers', {}).items()
                             if k in ('clip', 'pickscore')}
    for options in [evaluation, *evaluation['scorers'].values()]:
        for key in ('python', 'checkpoint', 'repo'):
            if key in options:
                value = options[key]
                if key != 'python' or '/' in value or '\\' in value:
                    options[key] = str(resolve(source.parent, value))
    if output.exists() and read_json(output) != config:
        raise ValueError(f'{output} already exists with different settings; choose another --out')
    provenance = dict(purpose='development, not a full paper benchmark', stage=stage,
                      source_sha256=file_digest(manifest_path), selection_seed=0,
                      source_provenance=manifest.get('provenance', {}))
    save_manifest(subset_path, 'coco', selected, provenance)
    write_json(output, config)
    plan, _ = load_config(output)
    from cfgctrl.benchmark.config import cells, samples
    total = sum(sum(1 for _ in samples(plan, cell)) for cell in cells(plan))
    print(f'{stage}: {count} prompts, {len(seeds)} seeds, {len(config["arms"])} arms; {total} images')
    print(f'Configuration: {output}')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('configs/paper.json'))
    parser.add_argument('--stage', choices=STAGES, default='dev')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', help='CFG, paper, and the candidate arm specifications')
    args = parser.parse_args()
    prepare(args.source, args.out, args.stage, args.arms)
