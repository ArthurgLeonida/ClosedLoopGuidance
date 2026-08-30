# Provenance of the vendored `dc/` package

`dc/` is a **byte-identical copy** of the guidance implementation from the
author's undergraduate thesis. It is vendored, not reimplemented, and it must
stay that way.

## Why vendored rather than rewritten

The central claim of this project is *"same controller, same gains, two plants,
no retuning."* If Plant A's tangential decomposition differs from Plant B's by
a sign, a normalisation, or which branch is amplified, then any difference
measured between the two plants is confounded with an implementation
difference, and the transfer result means nothing. Those bugs are silent: they
produce plausible gradients that are quietly wrong.

So the guidance primitives are copied unchanged, and the control layer is
injected by **subclassing** (`plants/plant_a_latent.py::ControlledDC` overrides
`_get_current_stg_scale`). No file under `dc/` is edited.

## Source

| field | value |
|---|---|
| repository | https://github.com/ArthurgLeonida/DreamCatalyst-PFC |
| commit | `eaa98aa56d7040461978242a375483afa074ee1a` |
| date | 2026-08-18 |
| source path | `nerfstudio/dc/` |
| copied | 2026-08-30 |

### File hashes at time of copy

Verify with `git hash-object <file>` in either repository.

```
4c348b5c7c7bddfe756f19575a1b7372b103f3a1  dc.py
93d679ea4861132656fcb118bd3f69c664d67710  dc_unet.py
fa515679c17b026e87ecce6ae294c734acec7ce6  guidance_utils.py
6f5d558bcb52c1c7be7baca36eb8de4b35b4f787  attention_utils.py
306857b7d5985bcc20de51adfe2fb170db3b6fc3  localization_utils.py
```

`dc/utils/` was copied wholesale because `dc.py` imports `free_lunch` and
`logging_utils` from it.

## What was deliberately NOT copied

| file | lines | why |
|---|---|---|
| `mask_voxel_cache.py` | 868 | 3D only. Enters `dc.py` from the caller via the `external_grad_mask*` arguments, so excluding it is safe; Plant A passes the defaults. |
| `method_config.py` | 116 | nerfstudio method registration. |

## Drift check, before the Plant B transfer experiment

The transfer claim requires that both plants really did share an
implementation. Before running Plant B, confirm the copies have not diverged:

```bash
for f in dc.py dc_unet.py guidance_utils.py attention_utils.py localization_utils.py; do
  cmp -s "$THESIS_REPO/nerfstudio/dc/$f" "dc/$f" \
    && echo "identical  $f" || echo "DIFFERS    $f"
done
```

If anything differs, reconcile it *before* running, and record in the write-up
that the two plants were verified to share one implementation.

## Licensing: UNRESOLVED, action required

Neither this repository nor `DreamCatalyst-PFC` currently contains a `LICENSE`
file, and `dc/` is derived from DreamCatalyst's official released code, itself
built on nerfstudio (Apache-2.0).

Before making this repository public:

1. Check DreamCatalyst's license terms and whether redistribution of derived
   code is permitted and under what conditions.
2. Carry the required attribution and notices into this repository.
3. Add a `LICENSE` here, and to `DreamCatalyst-PFC`, which is already public
   and already linked from graduate applications. A public repository with no
   license grants no rights to anyone who clones it.

Upstream references:

- DreamCatalyst, ICLR 2025 — https://arxiv.org/abs/2407.11394
- TAG, ICML 2026 — https://arxiv.org/abs/2510.04533
- STG, CVPR 2025 — https://arxiv.org/abs/2411.18664
- nerfstudio — Apache-2.0
