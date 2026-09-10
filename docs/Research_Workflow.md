# Developing a controller before the full benchmark

Use small paired experiments while the method is changing. The full paper matrix is a final evaluation configuration; running all 121,200 images for each idea spends most of the budget before answering whether the idea helps.

The research preparation script copies one model's settings (SD3.5 by default) and CLIP/PickScore evaluator paths from your existing configuration. It selects complete prompt records from the frozen COCO manifest without choosing new captions. Every method receives identical prompts and seeds. It leaves the full configuration and existing images intact.

| Stage | Prompts | Seeds | Default methods | Images |
|---|---:|---:|---:|---:|
| smoke | 8 | 1 | 4 | 32 |
| dev | 64 | 1 | 4 | 256 |
| validation | 256 | 2 | 4 | 2,048 |

The defaults compare CFG, the paper controller, proximal and proximal_relative. Smoke uses eight of the development prompts. Validation prompts are disjoint from development prompts for the same source manifest. Counts assume one guidance scale, as in the current paper configuration.

## Run an iteration

First stop any full generation run that is using the GPUs you want, and retain its output. Run from the repository root after syncing the new script:

~~~bash
nohup bash -c '
  set -e
  export CLG_ENV=/home/jovyan/compartilhado/envs/clg-generation
  source ./vlab_env.sh
  unset CUDA_VISIBLE_DEVICES
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  python -u scripts/prepare_research.py \
    --source configs/paper.json --stage dev \
    --out configs/research_dev.json
  bash scripts/run_benchmarks.sh \
    --config configs/research_dev.json \
    --out results/research_dev_v1 \
    --gpus 0 2 5
' > research_dev_v1.nohup.log 2>&1 &
~~~

The same wrapper collects generation logs, per-image controller traces, diagnostics.csv, CLIP/PickScore results, reports and GPU telemetry. Only these two selected metrics are required; the small profile does not need CompBench or MPS. Your source configuration must already point to the installed general metrics environment.

Use --stage smoke with different config/output paths for a first implementation check. Use --stage validation with separate paths after selecting a candidate. Keep validation out of the tuning loop; repeated selection on that split makes it another development set.

For focused ablations, preparation accepts --arms, for example:
`--arms cfg paper relative005=proximal_relative:k=0.05 relative01=proximal_relative:k=0.1`.
Keep CFG and paper references, change one mechanism at a time, and compare tuning budgets fairly. The current methods are candidates and baselines; their presence does not establish novelty.

Keep steps and resolution at their source values initially (30 and 1024 in paper.json). Reducing these can be useful for debugging, but it also changes the dynamics and may change method rankings. Reducing prompts, models and redundant arms is the first cost reduction.

## One method or a small gain sweep

Use `--from-run` to copy the exact COCO manifest, seeds, model settings and evaluator paths from a completed run, including any earlier manual changes. This option replaces `--source`/`--stage` selection; it does not regenerate baselines or import their images into the new directory. The `--model` flag selects a single backbone; `--arms` selects controller variants. GPU count controls how those images are distributed, independently of model/arm count.

After syncing the updated scripts, this generates only 64 relative-proximal images at k=0.3 and compares their existing metric scores against CFG and paper from the first development run:

~~~bash
nohup bash -c '
  set -e
  export CLG_ENV=/home/jovyan/compartilhado/envs/clg-generation
  source ./vlab_env.sh
  unset CUDA_VISIBLE_DEVICES
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  python -u scripts/prepare_research.py \
    --from-run results/research_dev_v1 --model sd35 \
    --arms rel03=proximal_relative:k=0.3 \
    --out configs/research_rel03.json
  bash scripts/run_benchmarks.sh \
    --config configs/research_rel03.json \
    --out results/research_rel03 --gpus 0 2 5
  python -u scripts/compare_research.py \
    --reference results/research_dev_v1 \
    --candidate results/research_rel03 \
    --out results/research_rel03/paired_comparison.json
' > research_rel03.nohup.log 2>&1 &
~~~

For a three-gain sweep, replace the arm list with `--arms rel03=proximal_relative:k=0.3 rel06=proximal_relative:k=0.6 rel10=proximal_relative:k=1.0`, and use new config, run, comparison and log paths such as `research_rel_sweep`. This makes 192 images with the original 64-prompt/one-seed design. Reuse the existing k=0.1 result as the initial reference point. All variants need distinct names because each name determines its output folder. To include baselines in a fresh run, add `cfg paper` to the arm list; this adds 128 images.

For another model, use `--source configs/paper.json --stage dev --model flux` (or qwen) in the preparation command, with separate output paths. A frozen SD3.5-only run cannot supply an absent model's settings. The resulting model configuration may be edited before its first generation; changes to steps, scales, resolution or seeds require separate comparisons from the original SD3.5 experiment.

The comparison script reads existing per-image metric artifacts, verifies complete image/record coverage and current metric fingerprints, checks matching generation settings/implementation/environments and evaluator metadata, then compares every candidate against CFG and paper at the same guidance scale. It writes paired mean differences, prompt win/tie rates, and 95% bootstrap intervals. Multiple seeds are averaged within each prompt before resampling prompts; the intervals describe prompt variability conditional on the sampled seeds, not all possible seed variability. The intervals are exploratory and are not corrected for testing multiple gains. Missing baselines or changed settings fail explicitly. This script supports CLIP and PickScore and does not run metric inference.

For a first k ablation, keep the saved paper controller at k=0.1, lambda=6 and CFG at w=7.5 fixed. Once a candidate is selected, also compare fairly tuned baselines on the development set and evaluate the selected settings on the reserved validation prompts. Gain/scale tuning budgets and the selection metric should be declared in advance. Unchanged baseline images can be reused for analysis, but baseline results from different prompts, seeds, samplers, resolutions or environments are not interchangeable.

## Interpreting a larger relative gain

Relative proximal uses `threshold = k * rms(e)`, then soft-thresholds the CFG extrapolation. A larger k removes more extrapolation at a fixed measured error. k=0 recovers CFG; sufficiently large thresholds remove extrapolation altogether and give the conditional prediction at that step. Thus a larger k can help or hurt alignment and quality. The underlying fixed-threshold operator is standard [soft-thresholding](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf); a quality improvement from this adaptive use remains an experimental question.

Start with k=0.3; if compute permits, bracket it with 0.6 and 1.0. These are exploratory values, not established optima. Relative k is dimensionless; the same numeric k in the paper controller has a different meaning. Do not tune it merely to match the paper's correction amplitude or surface RMS. Keep guidance fixed for this initial sweep, then separately test whether any gain is explained by reducing ordinary CFG guidance. Inspect the same images and use additional metrics on promising existing outputs before claiming a method improvement.

## What to measure and when to expand

1. Check that the intended mechanism works: correction magnitude, late switching, actual velocity changes after dtype conversion, and runtime. A smaller surface RMS alone is not evidence of better generated images.
2. Inspect the same prompt/seed images side by side and compare paired CLIP/PickScore changes. These are screening signals; a small mean increase on 64 prompts is not a reliable final result.
3. Validate a selected candidate on the reserved prompts and multiple seeds. Quantify uncertainty in paired differences, resampling prompts together with their seeds. The per-run report gives scores; `scripts/compare_research.py` adds the paired analysis described above.
4. Test transfer on another backbone and targeted compositional prompts before the full benchmark. Use attribute/spatial or VQAScore evaluations earlier if the claimed improvement specifically concerns those capabilities; COCO screening alone cannot test that claim.
5. Once the method and hyperparameters are fixed, run the declared full benchmark metrics for CFG, the paper method and the selected candidate, then the necessary ablations. Six arms are not required for every final model/benchmark combination.

FID is omitted from the tiny profiles. Report full benchmark FID later using the declared reference protocol. For main claims, report uncertainty and what sources of variability it captures, consistent with the [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist).

These profiles are subsets of the existing COCO benchmark, so the final full COCO set overlaps development. Disclose this tuning overlap and report an untouched complement or use a separate development dataset before claiming held-out generalization. The validation subset is held out from this development profile, not from all prior research automatically.

## Preserve earlier work

Do not repoint results/paper_all to the small configuration. Every changed method, prompt set, seed set or generation implementation needs a fresh output directory. Unchanged runs resume normally.

Previously generated full-run images remain useful for inspection and later comparisons, but the current cache fingerprints include the entire generation plan and implementation. They are not automatically imported into the small run, and controller source changes can prevent resuming the old full run. Keep the original code/configuration revision for that run. Small runs intentionally use their own outputs to avoid mislabeling images produced by different methods.
