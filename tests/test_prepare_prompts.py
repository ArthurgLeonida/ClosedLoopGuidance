"""Exercise the COCO prompt preparation without downloading COCO."""

import json

import pytest

from data import prepare_prompts


def coco(tmp_path, n_images=20, captions_per_image=3):
    """A COCO-shaped caption file: several captions per image, ids out of order."""
    images = [{"id": 1000 + i, "file_name": f"{1000 + i:012d}.jpg"} for i in range(n_images)]
    annotations = []
    ann_id = 5000
    for i in range(n_images):
        for c in range(captions_per_image):
            annotations.append({"image_id": 1000 + i, "id": ann_id + c * 7,
                                "caption": f"caption {c} for image {i}"})
        ann_id += 1
    annotations.reverse()                       # order must not matter
    path = tmp_path / "captions.json"
    path.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")
    return path


def test_one_caption_per_image_chosen_deterministically(tmp_path):
    pairs = prepare_prompts.load_pairs(coco(tmp_path))
    assert len(pairs) == 20
    assert [p[0] for p in pairs] == sorted(p[0] for p in pairs)      # sorted by image id
    # the lowest annotation id wins, which is caption 0 here
    assert all(p[2].startswith("caption 0 ") for p in pairs)
    assert prepare_prompts.load_pairs(coco(tmp_path)) == pairs        # reproducible


def test_captions_are_whitespace_normalised(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [{"image_id": 1, "id": 1, "caption": "  a  man\n  on a\tbike \n"}]}),
        encoding="utf-8")
    assert prepare_prompts.load_pairs(path)[0][2] == "a man on a bike"


def test_empty_and_malformed_inputs_are_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"images": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="has no 'annotations'"):
        prepare_prompts.load_pairs(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"images": [], "annotations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no usable image/caption pairs"):
        prepare_prompts.load_pairs(empty)


def test_tune_and_test_are_disjoint_and_reproducible(tmp_path):
    pairs = prepare_prompts.load_pairs(coco(tmp_path))
    tune, test = prepare_prompts.split(pairs, n_tune=5, n_test=7, seed=0)
    assert len(tune) == 5 and len(test) == 7
    assert not ({r[0] for r in tune} & {r[0] for r in test})
    assert prepare_prompts.split(pairs, 5, 7, seed=0) == (tune, test)
    assert prepare_prompts.split(pairs, 5, 7, seed=1) != (tune, test)   # seed matters


def test_split_refuses_to_overdraw(tmp_path):
    pairs = prepare_prompts.load_pairs(coco(tmp_path, n_images=10))
    with pytest.raises(ValueError, match="only 10 images are available"):
        prepare_prompts.split(pairs, n_tune=8, n_test=8, seed=0)


def test_written_prompt_file_is_one_caption_per_line(tmp_path):
    pairs = prepare_prompts.load_pairs(coco(tmp_path, n_images=4))
    out = tmp_path / "p.txt"
    prepare_prompts.write_prompts(out, pairs)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [p[2] for p in pairs]
    # and real_model.py reads it back the same way
    parsed = [l.strip() for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert parsed == [p[2] for p in pairs]
