# Benchmark data (not committed)

## PIE-Bench

From PnPInversion (ICLR 2024): https://cure-lab.github.io/PnPInversion/

700 images across 10 editing types. Each entry carries a source prompt, a
target prompt, an editing instruction, edit subjects, and an editing mask.

Expected layout:

    bench/pie_bench/
        annotation_images/
        mapping_file.json
        evaluation/            <- their metric script

## Development subset

Do NOT develop on all 700. Stratify 5 images from each of the 10 edit types =
50 pairs. The whole hypothesis is that the right guidance differs by edit type,
so the subset must span that axis rather than being sampled at random. Scale to
the full set only for final numbers on the arms that survive.

Write the chosen subset to `bench/subset_50.json` and commit that file, so
every experiment is reproducible against the same pairs.
