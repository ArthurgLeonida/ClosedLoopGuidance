"""Compare new controller arms with saved baselines using paired prompt scores.

Reads existing images and metric artifacts; no generation or model inference.
Bootstrap intervals are exploratory, not adjusted for hyperparameter selection.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfgctrl.benchmark.common import read_json, write_json
from cfgctrl.benchmark.config import load_plan
from cfgctrl.benchmark.evaluation import aggregate_scores, prepare_jobs


def paired_statistics(reference, candidate, prompt_ids, resamples=10000):
    import numpy as np
    if len(reference) != len(candidate) or len(reference) != len(prompt_ids):
        raise ValueError('Paired scores must have equal coverage')
    groups = defaultdict(list)
    for prompt, a, b in zip(prompt_ids, reference, candidate):
        groups[prompt].append((float(a), float(b)))
    if len(groups) < 2 or resamples < 100:
        raise ValueError('Need at least two prompts and 100 bootstrap resamples')
    means = np.asarray([np.mean(groups[key], axis=0) for key in sorted(groups)])
    if not np.isfinite(means).all():
        raise ValueError('Nonfinite paired scores')
    delta = means[:, 1] - means[:, 0]
    rng = np.random.default_rng(0)
    # Chunk sampling to keep memory bounded for larger validation sets.
    boot = []
    for start in range(0, resamples, 256):
        indices = rng.integers(len(delta), size=(min(256, resamples - start), len(delta)))
        boot.extend(delta[indices].mean(axis=1).tolist())
    return dict(n_prompts=len(groups), n_images=len(prompt_ids),
                reference_mean=float(means[:, 0].mean()), candidate_mean=float(means[:, 1].mean()),
                mean_difference=float(delta.mean()),
                bootstrap_95_ci=np.percentile(boot, [2.5, 97.5]).tolist(),
                prompt_win_rate=float((delta > 0).mean()),
                prompt_tie_rate=float((delta == 0).mean()),
                prompt_differences=dict(zip(sorted(groups), delta.tolist())))


def check_design(reference, candidate, model):
    if reference['models'][model] != candidate['models'][model]:
        raise ValueError('Model settings differ; this comparison isolates controller changes')
    if reference['generation_implementation'] != candidate['generation_implementation']:
        raise ValueError('Generation implementation differs; audit the code before comparing')
    a, b = reference['benchmarks']['coco'], candidate['benchmarks']['coco']
    if (a['kind'] != 'coco' or b['kind'] != 'coco' or
            sorted(a['seeds']) != sorted(b['seeds']) or
            sorted(a['manifest']['records'], key=lambda r: r['id']) !=
            sorted(b['manifest']['records'], key=lambda r: r['id'])):
        raise ValueError('Prompts/captions/seeds differ; use --from-run when preparing the candidate')


def read_scores(task, metric):
    artifact = read_json(task['output'])
    if artifact.get('input_fingerprint') != task['input_fingerprint']:
        raise ValueError(f'Stale metric artifact: {task["output"]}; evaluate it again')
    aggregate_scores(metric, artifact['scores'], task['records'])
    return artifact, {r['id']: r['score'] for r in artifact['scores']}


def compare(reference_dir, candidate_dir, reference_arms, metrics, model, resamples=10000):
    reference, ref_base, ref_hash = load_plan(reference_dir)
    candidate, new_base, new_hash = load_plan(candidate_dir)
    check_design(reference, candidate, model)
    env_a = read_json(Path(reference_dir) / 'generation_environment.json')
    env_b = read_json(Path(candidate_dir) / 'generation_environment.json')
    if env_a != env_b:
        raise ValueError('Recorded generation environments differ; audit them before comparing')
    ref_jobs = prepare_jobs(reference_dir, reference, ref_hash, ref_base,
                            models=[model], benchmarks=['coco'], metrics=metrics)
    new_jobs = prepare_jobs(candidate_dir, candidate, new_hash, new_base,
                            models=[model], benchmarks=['coco'], metrics=metrics)
    rows = []
    for metric in metrics:
        references = {(t['cell']['arm'], t['cell']['w']): t for t in ref_jobs[metric]}
        for task in new_jobs[metric]:
            artifact_b, scores_b = read_scores(task, metric)
            for arm in reference_arms:
                key = (arm, task['cell']['w'])
                if key not in references:
                    raise ValueError(f'No reference cell for {key}')
                ref_task = references[key]
                artifact_a, scores_a = read_scores(ref_task, metric)
                for field in ('metric', 'protocol', 'versions', 'resource'):
                    if artifact_a.get(field) != artifact_b.get(field):
                        raise ValueError(f'Evaluator {field} differs for {metric}')
                ids = [r['id'] for r in task['records']]
                if set(ids) != set(scores_a):
                    raise ValueError('Reference and candidate sample IDs differ')
                stats = paired_statistics([scores_a[i] for i in ids], [scores_b[i] for i in ids],
                                          [r['record']['id'] for r in task['records']], resamples)
                row = dict(model=model, benchmark='coco', metric=metric, w=task['cell']['w'],
                           reference_arm=arm, candidate_arm=task['cell']['arm'], **stats)
                rows.append(row)
                print(f'{row["candidate_arm"]} vs {arm}, {metric}: '
                      f'delta={stats["mean_difference"]:+.6f}, '
                      f'95% CI={stats["bootstrap_95_ci"]}, wins={stats["prompt_win_rate"]:.1%}')
    return dict(reference_run=str(Path(reference_dir).resolve()),
                candidate_run=str(Path(candidate_dir).resolve()),
                interval='paired prompt bootstrap; seeds averaged within prompt; exploratory, no multiplicity adjustment',
                resamples=resamples, bootstrap_seed=0, results=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference-arms', nargs='+', default=['cfg', 'paper'])
    parser.add_argument('--metrics', nargs='+', choices=['clip', 'pickscore'], default=['clip', 'pickscore'])
    parser.add_argument('--model', default='sd35')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.reference_arms,
                     args.metrics, args.model, args.resamples)
    write_json(args.out, result)
