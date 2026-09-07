"""The fidelity half of the image-quality pipeline.

clean-fid is not exercised here: a fake backend is injected instead, so these
tests cover what this repository is responsible for -- finding the right
directories, refusing comparisons that sample size would decide, resuming, and
writing a CSV that pareto.py can actually read.
"""

import argparse
import csv
import json

import pytest

from experiments import fid, pareto


class FakeBackend:
    """Records how it was called and returns a value derived from the path, so
    a test can tell which directory produced which row."""

    def __init__(self):
        self.calls = []

    def _value(self, fdir1, kwargs):
        self.calls.append((fdir1, dict(kwargs)))
        return float(len(str(fdir1)))

    def compute_fid(self, fdir1, **kwargs):
        return self._value(fdir1, kwargs)

    def compute_kid(self, fdir1, **kwargs):
        return self._value(fdir1, kwargs) / 1000.0


def make_run(tmp_path, arms=("cfg", "paper"), ws=(1.5, 3.0), seeds=(0,),
             prompts=("a red cube", "a blue sphere"), skip=()):
    run = tmp_path / "run"
    (run).mkdir(exist_ok=True)
    (run / "config.json").write_text(json.dumps({
        "prompts": list(prompts), "arms": list(arms),
        "w": list(ws), "seeds": list(seeds),
    }), encoding="utf-8")
    for arm in arms:
        for w in ws:
            cell = run / arm / f"w{w}"
            cell.mkdir(parents=True, exist_ok=True)
            for pid in range(len(prompts)):
                for seed in seeds:
                    if (arm, w, pid, seed) in skip:
                        continue
                    (cell / f"p{pid:02d}_s{seed}.png").write_bytes(b"x")
    return run


def compute_args(run, reference, **over):
    args = argparse.Namespace(
        run=str(run), reference=str(reference), out="fid.csv", device="cpu",
        mode="clean", batch=8, workers=0, kid=True, cache_stats=True,
        stats_name="test_ref", refresh_stats=False, allow_unequal=False,
        resume=False, dry_run=False, backend=FakeBackend())
    for key, value in over.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def reference(tmp_path):
    ref = tmp_path / "ref"
    ref.mkdir()
    for i in range(4):
        (ref / f"r{i}.jpg").write_bytes(b"x")
    return ref


# ------------------------------------------------------------------ discovery

def test_cells_are_one_directory_per_arm_and_scale(tmp_path):
    run = make_run(tmp_path)
    cfg = fid.load_run(run)
    dirs, missing = fid.discover_cells(run, cfg)
    assert not missing
    assert set(dirs) == {("cfg", 1.5), ("cfg", 3.0), ("paper", 1.5), ("paper", 3.0)}
    assert dirs[("paper", 3.0)] == run / "paper" / "w3.0"


def test_only_image_files_are_counted(tmp_path):
    run = make_run(tmp_path)
    cell = run / "cfg" / "w1.5"
    (cell / "notes.txt").write_text("not an image", encoding="utf-8")
    assert fid.count_images(cell) == 2


# ------------------------------------------------------- sample-size guarding

def test_unequal_cells_are_refused(tmp_path, reference, capsys):
    """A cell that lost images to a killed job scores better for a reason that
    has nothing to do with the guidance law."""
    run = make_run(tmp_path, skip={("paper", 3.0, 1, 0)})
    with pytest.raises(ValueError, match="different image counts"):
        fid.cmd_compute(compute_args(run, reference))


def test_unequal_cells_can_be_forced(tmp_path, reference, capsys):
    run = make_run(tmp_path, skip={("paper", 3.0, 1, 0)})
    assert fid.cmd_compute(compute_args(run, reference, allow_unequal=True)) == 0
    assert "WARNING" in capsys.readouterr().out


def test_dry_run_computes_nothing(tmp_path, reference):
    run = make_run(tmp_path)
    args = compute_args(run, reference, dry_run=True)
    assert fid.cmd_compute(args) == 0
    assert args.backend.calls == []
    assert not (run / "fid.csv").exists()


# ----------------------------------------------------------------- scoring

def test_every_cell_is_scored_against_the_reference(tmp_path, reference):
    run = make_run(tmp_path)
    args = compute_args(run, reference)
    assert fid.cmd_compute(args) == 0
    scored = {call[0] for call in args.backend.calls}
    assert scored == {str(run / arm / f"w{w}")
                      for arm in ("cfg", "paper") for w in (1.5, 3.0)}
    assert all(call[1]["fdir2"] == str(reference) for call in args.backend.calls)


def test_cached_reference_statistics_replace_the_directory(tmp_path, reference):
    """With stats cached, the reference images must not be walked again."""
    run = make_run(tmp_path)
    backend = FakeBackend()
    value, _ = fid.score_cell(backend, run / "cfg" / "w1.5", reference, "test_ref",
                              "clean", "cpu", 0, 8, want_kid=False)
    assert value > 0
    assert backend.calls[0][1]["dataset_name"] == "test_ref"
    assert backend.calls[0][1]["dataset_split"] == "custom"
    assert "fdir2" not in backend.calls[0][1]


def test_kid_can_be_skipped(tmp_path, reference):
    run = make_run(tmp_path)
    args = compute_args(run, reference, kid=False)
    fid.cmd_compute(args)
    assert len(args.backend.calls) == 4          # not 8: no second pass
    rows = list(csv.DictReader(open(run / "fid.csv", newline="", encoding="utf-8")))
    assert all(row["kid"] == "" for row in rows)


def test_a_failing_kid_does_not_lose_the_fid(tmp_path, reference, capsys):
    class NoKid(FakeBackend):
        def compute_kid(self, fdir1, **kwargs):
            raise RuntimeError("this clean-fid has no custom KID stats")

    run = make_run(tmp_path)
    args = compute_args(run, reference, backend=NoKid())
    assert fid.cmd_compute(args) == 0
    assert "KID unavailable" in capsys.readouterr().out
    rows = list(csv.DictReader(open(run / "fid.csv", newline="", encoding="utf-8")))
    assert len(rows) == 4 and all(float(row["fid"]) > 0 for row in rows)


# ------------------------------------------------------------------- resuming

def test_resume_keeps_finished_cells(tmp_path, reference):
    run = make_run(tmp_path)
    first = compute_args(run, reference)
    fid.cmd_compute(first)
    before = (run / "fid.csv").read_text(encoding="utf-8")

    second = compute_args(run, reference, resume=True)
    assert fid.cmd_compute(second) == 0
    assert second.backend.calls == []                       # nothing recomputed
    assert (run / "fid.csv").read_text(encoding="utf-8") == before


def test_without_resume_every_cell_is_recomputed(tmp_path, reference):
    run = make_run(tmp_path)
    fid.cmd_compute(compute_args(run, reference))
    again = compute_args(run, reference)
    fid.cmd_compute(again)
    assert len(again.backend.calls) == 8                    # 4 cells, FID + KID


def test_partial_results_survive_a_crash(tmp_path, reference):
    """The CSV is rewritten after every cell, so a job killed at cell 3 of 4
    can be resumed rather than restarted."""
    class DiesOnThirdCell(FakeBackend):
        cells = 0

        def compute_fid(self, fdir1, **kwargs):
            self.cells += 1
            if self.cells > 2:
                raise KeyboardInterrupt
            return super().compute_fid(fdir1, **kwargs)

    run = make_run(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        fid.cmd_compute(compute_args(run, reference, backend=DiesOnThirdCell()))
    rows = list(csv.DictReader(open(run / "fid.csv", newline="", encoding="utf-8")))
    assert 0 < len(rows) < 4


# ----------------------------------------------------- hand-off to pareto.py

@pytest.mark.parametrize("column", ["fid", "kid"])
def test_the_csv_is_what_pareto_reads(tmp_path, reference, column):
    """The contract between the two scripts, checked rather than assumed."""
    run = make_run(tmp_path)
    fid.cmd_compute(compute_args(run, reference))
    values = pareto.read_fidelity(run / "fid.csv", column)
    assert set(values) == {("cfg", 1.5), ("cfg", 3.0), ("paper", 1.5), ("paper", 3.0)}
    assert all(v > 0 for v in values.values())


def test_a_blank_kid_column_fails_loudly(tmp_path, reference):
    """--no-kid leaves the column present but empty. Joining on it must say so
    rather than quietly ranking arms on no data."""
    run = make_run(tmp_path)
    fid.cmd_compute(compute_args(run, reference, kid=False))
    with pytest.raises(ValueError, match="no usable rows"):
        pareto.read_fidelity(run / "fid.csv", "kid")


# ----------------------------------------------------- the matched reference

def write_pairs(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wri = csv.writer(fh)
        wri.writerow(["image_id", "file_name", "caption"])
        wri.writerows(rows)
    return path


def reference_args(tmp_path, run=None, **over):
    args = argparse.Namespace(
        images=str(tmp_path / "val2017"), run=(str(run) if run else None),
        pairs=str(tmp_path / "pairs.csv"), out=str(tmp_path / "matched"))
    for key, value in over.items():
        setattr(args, key, value)
    return args


def test_captions_are_matched_after_collapsing_whitespace(tmp_path):
    write_pairs(tmp_path / "pairs.csv", [(1, "a.jpg", "A red  cube\non grass")])
    assert fid.read_pairs(tmp_path / "pairs.csv") == {"A red cube on grass": "a.jpg"}


def test_the_subset_holds_one_image_per_prompt(tmp_path):
    src = tmp_path / "val2017"
    src.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (src / name).write_bytes(b"x")
    write_pairs(tmp_path / "pairs.csv",
                [(1, "a.jpg", "a red cube"), (2, "b.jpg", "a blue sphere"),
                 (3, "c.jpg", "an unused caption")])
    run = make_run(tmp_path)

    assert fid.cmd_reference(reference_args(tmp_path, run)) == 0
    out = tmp_path / "matched"
    assert sorted(p.name for p in out.iterdir()) == ["a.jpg", "b.jpg"]


def test_prompts_absent_from_the_pairs_file_are_refused(tmp_path):
    """Silent failure to guard: an unmatched reference still produces a number."""
    src = tmp_path / "val2017"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"x")
    write_pairs(tmp_path / "pairs.csv", [(1, "a.jpg", "a red cube")])
    run = make_run(tmp_path, prompts=("a red cube", "a caption from another set"))
    with pytest.raises(ValueError, match="not in"):
        fid.cmd_reference(reference_args(tmp_path, run))


def test_a_matched_image_missing_from_the_archive_is_refused(tmp_path):
    src = tmp_path / "val2017"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"x")            # b.jpg never unzipped
    write_pairs(tmp_path / "pairs.csv",
                [(1, "a.jpg", "a red cube"), (2, "b.jpg", "a blue sphere")])
    run = make_run(tmp_path)
    with pytest.raises(ValueError, match="not in"):
        fid.cmd_reference(reference_args(tmp_path, run))


def test_without_a_run_nothing_is_subset(tmp_path, capsys):
    src = tmp_path / "val2017"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"x")
    assert fid.cmd_reference(reference_args(tmp_path)) == 0
    assert not (tmp_path / "matched").exists()
    assert "1 images" in capsys.readouterr().out
