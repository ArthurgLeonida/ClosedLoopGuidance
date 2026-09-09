# Developing a controller before the full benchmark

Use small paired experiments while the method is changing. The full paper matrix is a final evaluation configuration; running all 121,200 images for each idea spends most of the budget before answering whether the idea helps.

The research preparation script copies the SD3.5 settings and CLIP/PickScore evaluator paths from your existing configuration. It selects complete prompt records from the frozen COCO manifest without choosing new captions. Every method receives identical prompts and seeds. It leaves the full configuration and existing images intact.

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

## What to measure and when to expand

1. Check that the intended mechanism works: correction magnitude, late switching, actual velocity changes after dtype conversion, and runtime. A smaller surface RMS alone is not evidence of better generated images.
2. Inspect the same prompt/seed images side by side and compare paired CLIP/PickScore changes. These are screening signals; a small mean increase on 64 prompts is not a reliable final result.
3. Validate a selected candidate on the reserved prompts and multiple seeds. Quantify uncertainty in paired differences, resampling prompts together with their seeds. The existing report gives scores; it does not automatically provide this paired uncertainty analysis.
4. Test transfer on another backbone and targeted compositional prompts before the full benchmark. Use attribute/spatial or VQAScore evaluations earlier if the claimed improvement specifically concerns those capabilities; COCO screening alone cannot test that claim.
5. Once the method and hyperparameters are fixed, run the declared full benchmark metrics for CFG, the paper method and the selected candidate, then the necessary ablations. Six arms are not required for every final model/benchmark combination.

FID is omitted from the tiny profiles. Report full benchmark FID later using the declared reference protocol. For main claims, report uncertainty and what sources of variability it captures, consistent with the [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist).

These profiles are subsets of the existing COCO benchmark, so the final full COCO set overlaps development. Disclose this tuning overlap and report an untouched complement or use a separate development dataset before claiming held-out generalization. The validation subset is held out from this development profile, not from all prior research automatically.

## Preserve earlier work

Do not repoint results/paper_all to the small configuration. Every changed method, prompt set, seed set or generation implementation needs a fresh output directory. Unchanged runs resume normally.

Previously generated full-run images remain useful for inspection and later comparisons, but the current cache fingerprints include the entire generation plan and implementation. They are not automatically imported into the small run, and controller source changes can prevent resuming the old full run. Keep the original code/configuration revision for that run. Small runs intentionally use their own outputs to avoid mislabeling images produced by different methods.
