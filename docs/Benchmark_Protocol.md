# Benchmark protocol and comparison limits

This document defines what the pipeline measures. The target is [CFG-Ctrl, main Table 2 and supplementary Tables 4–5](https://arxiv.org/html/2603.03281v2), plus controlled comparisons of this repository's implemented fixes.

## Generation

| Family | Checkpoint | True CFG scale | Steps | Resolution | Paper gain |
|---|---|---:|---:|---|---:|
| SD3.5 | stabilityai/stable-diffusion-3.5-large | 7.5 | 30 | 1024² | 0.1 |
| FLUX | black-forest-labs/FLUX.1-dev | 2.0 | 30 | 1024² | 0.7 |
| Qwen | Qwen/Qwen-Image | 4.0 | 30 | 1024² | 0.1 |

These are explicit local sampling settings. The paper specifies the models, resolution, λ=6 and controller gains, but does not completely specify sampling steps, seeds and every default used for each table. Treat the scale/step choices as part of our protocol. Pin checkpoint `revision` values to commit hashes before a long comparison; `main` is a mutable upstream reference.

The default arms are conditional, CFG, paper, smooth excess-only, proximal and relative proximal. Conditional uses one prediction per step at `w=1`. Guided arms share the same two prediction branches, initial seed and scheduler. FLUX's embedded guidance is fixed at 1 and kept separate from true CFG.

The adapters target Diffusers 0.35.2 and combine the predictions immediately before the scheduler step. The authors' released implementation uses DiffSynth; scheduler details and numerical results need not be identical. Qwen's native Diffusers norm rescaling is replaced with the raw CFG combination for **every** arm, matching the algebra of the authors' released controller combiner. Our Qwen CFG baseline therefore uses that explicit comparison convention.

The controller's raw correction and the actual velocity difference after casting to the scheduler input dtype are separate diagnostics. Euler displacement correction is the velocity difference multiplied by the sigma step length. Low-precision arithmetic can erase a sufficiently small correction.

## COCO and metric units

Select 5,000 unique images and one caption per image. This repository defaults to val2017 and selection seed 0; the authors have not released enough information to identify their exact subset/captions. `--pairs` accepts exact identities if they become available. FID reference images are selected from the manifest, never by taking arbitrary files from a directory.

| Metric | Local evaluator and reported scale | Better |
|---|---|---|
| FID | clean-fid, clean mode, Inception-v3 2048 features | Lower |
| CLIP | OpenAI CLIP ViT-L/14, normalized paired cosine | Higher |
| Aesthetic | Official improved-aesthetic-predictor L/14 linear MLP | Higher |
| ImageReward | Official ImageReward-v1.0 normalized reward | Higher |
| PickScore | PickScore_v1 normalized paired cosine | Higher |
| HPSv2 | Official HPS_v2_compressed checkpoint, cosine | Higher |
| HPSv2.1 | Official HPS_v2.1_compressed checkpoint, cosine | Higher |
| MPS | Official overall checkpoint and overall condition, scaled cosine logit | Higher |

CLIP/PickScore/HPS are not multiplied by 100. PickScore is not softmaxed across prompts; MPS is not reduced to a one-image softmax. These choices are consistent with the scales of the published table, but that consistency is an inference: the paper does not release its full evaluator configuration.

FID is reported directly. Subtracting a real-versus-real “floor” does not produce the paper's FID and is not a valid general bias correction. With several generation seeds, this implementation pools all generated images against the same unique real set and reports both counts. Keep seed counts equal across arms; finite-sample FID changes with sample size.

Per-image metrics average seeds within each prompt, then average prompts. `prompt_sd` is dispersion across prompt means, not a confidence interval. The pipeline does not infer statistical significance or automatically select a winning controller.

Evaluator references: [clean-fid](https://github.com/GaParmar/clean-fid), [OpenAI CLIP](https://github.com/openai/CLIP), [aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor), [ImageReward](https://github.com/THUDM/ImageReward), [PickScore](https://github.com/yuvalkirstain/PickScore), [HPSv2](https://github.com/tgxs002/HPSv2), [MPS](https://github.com/Kwai-Kolors/MPS).

## Composition benchmarks

**T2I-CompBench:** the official validation color, shape, texture and spatial files contain 300 prompts each. Those are the four categories reported in the paper's supplement. The pipeline runs the official BLIP-VQA and UniDet spatial scripts at revision `4aa404212eb5d06e5adbcd9cee696c750d0d25a5`, verifies every question ID, and reports category means. It does not replace spatial detection with CLIP. The default is one image per prompt; the paper does not fully release its repetition protocol. [Official benchmark](https://github.com/Karine-Huang/T2I-CompBench).

**GenAI-Bench:** use the compositional text-to-image benchmark evaluated with VQAScore, not the unrelated benchmark for preference judges. The default is the released 1,600-prompt set, scored by CLIP-FlanT5-XXL through `t2v-metrics==1.1` with its original question and answer templates. Basic contains 722 prompts, Advanced 871; seven prompts have no skill tags. All images receive raw scores, while Overall follows the official evaluator's union of skill memberships (1,593). These counts were checked against the actual downloaded annotations. The default config applies this benchmark to SD3.5, as in the paper's supplementary comparison. [Dataset](https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600), [official evaluation code](https://github.com/linzhiqiu/t2v_metrics).

## Reproducibility and improvement claims

A saved run embeds its manifest and generation configuration. Image checksums, prompt/seed identities, controller traces and installed package versions accompany each image. Metric results record the scoring protocol, configuration, package versions, local weight hashes and external evaluator source identity. Changing evaluator settings can rescore the existing images; changing the generation plan requires a new run.

Mutable remote checkpoint defaults and unspecified paper settings still limit reproducibility. Pin available model/processor revisions, retain local evaluator weights and save environment package lists with an experiment. The pipeline records a model commit when the loader exposes it; it does not claim every upstream loader supplies a complete weight fingerprint.

Compare each proposed fix against retuned CFG and the paper controller, using paired prompts/seeds and a guidance-scale sweep. Removing a ±0.5 controller cycle is a verified mathematical improvement; it does not establish improved image quality. A gain at the same nominal scale can reflect weaker guidance. Compare quality at similar alignment and inspect composition/detail failures. Tune thresholds on separate prompts before final evaluation.

Other methods appearing in the original tables, such as CFG-Zero* and Rectified-CFG++, are outside the implemented arm set. This workflow evaluates the original CFG-Ctrl law and the local alternatives; it does not recreate every table baseline.

CPU tests exercise controller properties, model call contracts and pipeline recovery. Full pretrained-model generation and all heavyweight evaluator runs must be validated on the target GPU environment.
