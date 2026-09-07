"""Download required evaluator assets and exercise official CUDA inference.

The two example scores are setup diagnostics, never benchmark measurements.
Run through setup_compbench.sh in the dedicated evaluator environment.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from cfgctrl.benchmark.evaluation import compbench_command, parse_compbench_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A fresh directory prevents a previous result from hiding an inference failure.
    run = Path(tempfile.mkdtemp(prefix='smoke_', dir=out))
    import torch
    import spacy
    from detectron2.layers import nms_rotated
    assert torch.__version__.split('+')[0] == '2.5.1'
    assert torch.version.cuda == '12.4'
    assert torch.cuda.is_available(), 'CUDA is required'
    spacy.load('en_core_web_sm')
    boxes = torch.tensor([[0., 0., 10., 10., 0.]], device='cuda')
    kept = nms_rotated(boxes, torch.ones(1, device='cuda'), 0.5)
    assert kept.tolist() == [0], 'Detectron2 CUDA kernel failed'

    weight = repo / 'UniDet_eval/experts/expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth'
    weight.parent.mkdir(parents=True, exist_ok=True)
    if not weight.is_file():
        partial = weight.with_suffix('.pth.part')
        torch.hub.download_url_to_file(
            'https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/'
            'Unified_learned_OCIM_RS200_6x%2B2x.pth', str(partial))
        partial.replace(weight)
    examples = {
        'color': 'a green bench and a blue bowl_000000.png',
        'spatial': 'a horse on the right of a car_000003.png',
    }
    report = dict(kind='setup diagnostic, not benchmark', python=sys.executable,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), scores={},
                  packages={p: metadata.version(p) for p in
                            ('torchvision', 'transformers', 'timm', 'spacy', 'detectron2')})
    with weight.open('rb') as stream:
        report['unidet_weight_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    for category, filename in examples.items():
        directory = run / category
        samples = directory / 'samples'
        samples.mkdir(parents=True)
        prompt = filename.rsplit('_', 1)[0]
        shutil.copyfile(repo / 'examples/samples' / filename,
                        samples / f'{prompt}_000000.png')
        command, cwd, result = compbench_command(repo, category, directory, sys.executable)
        print(f'Running {category} inference; log: {directory / "inference.log"}', flush=True)
        with (directory / 'inference.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        report['scores'][category] = parse_compbench_results(result, [{'id': category}])[0]['score']
    (run / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'COMPBENCH_SMOKE_PASSED: {run}')


if __name__ == '__main__':
    main()
