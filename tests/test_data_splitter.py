"""Tests for cross-validation split generation and filesystem materialization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from utils.data_splitter import CrossValidationDataSplitter


def _create_mock_raw_dataset(raw_root: Path, base_ids: list[str]) -> None:
    """Create a minimal dataset tree expected by the splitter."""
    for time_name in ["before", "after"]:
        for data_type in ["image", "height"]:
            (raw_root / time_name / data_type).mkdir(parents=True, exist_ok=True)
    (raw_root / "mask").mkdir(parents=True, exist_ok=True)

    for base_id in base_ids:
        for sample_idx in range(1, 3):
            sample_stem = f"{base_id}-{sample_idx}"
            image = Image.new("RGB", (8, 8), color=(100, 150, 200))
            image.save(raw_root / "before" / "image" / f"{sample_stem}.png")
            image.save(raw_root / "after" / "image" / f"{sample_stem}.png")

            height = np.full((8, 8), fill_value=sample_idx, dtype=np.float32)
            np.save(raw_root / "before" / "height" / f"{sample_stem}.npy", height)
            np.save(raw_root / "after" / "height" / f"{sample_stem}.npy", height)
            np.save(raw_root / "mask" / f"{sample_stem}.npy", np.ones((8, 8), dtype=bool))


@pytest.fixture
def splitter_paths(tmp_path: Path) -> dict[str, object]:
    """Create source data and inhibitor metadata files for splitter tests."""
    base_ids = ["1-1-1", "1-1-2", "1-1-3", "1-1-4", "1-1-5", "1-1-6"]
    nacl_ids = {"1-1-1", "1-1-3"}

    source_dir = tmp_path / "raw"
    _create_mock_raw_dataset(raw_root=source_dir, base_ids=base_ids)

    inhibitor_list_file = tmp_path / "volume-loss.csv"
    inhibitor_rows = []
    for base_id in base_ids:
        inhibitor_name = "NaCl" if base_id in nacl_ids else "InhibitorA"
        inhibitor_rows.append({"id": f"{base_id}-1", "name": inhibitor_name})
    pd.DataFrame(inhibitor_rows).to_csv(inhibitor_list_file, index=False)

    return {
        "source_dir": source_dir,
        "target_dir": tmp_path / "splits",
        "inhibitor_list_file": inhibitor_list_file,
        "all_group_ids": set(base_ids),
        "nacl_group_ids": nacl_ids,
    }


def _build_splitter(
    splitter_paths: dict[str, object],
    random_seed: int,
    num_folds: int = 3,
    val_size: int = 1,
    test_size: int = 1,
) -> CrossValidationDataSplitter:
    """Construct a splitter with test-friendly defaults."""
    return CrossValidationDataSplitter(
        source_dir=splitter_paths["source_dir"],
        target_dir=splitter_paths["target_dir"],
        inhibitor_list_file=splitter_paths["inhibitor_list_file"],
        num_folds=num_folds,
        random_seed=random_seed,
        val_size=val_size,
        test_size=test_size,
    )


def test_extract_sample_group_id_handles_multi_digit_suffix() -> None:
    """Sample base IDs should be parsed from filenames with multi-digit suffixes."""
    assert CrossValidationDataSplitter._extract_sample_group_id("1-1-10-12") == "1-1-10"


def test_fold_generation_is_deterministic(splitter_paths: dict[str, object]) -> None:
    """A fixed seed should always produce identical fold membership."""
    splitter_a = _build_splitter(splitter_paths=splitter_paths, random_seed=11)
    splitter_b = _build_splitter(splitter_paths=splitter_paths, random_seed=11)

    assert splitter_a._train_folds == splitter_b._train_folds
    assert splitter_a._val_folds == splitter_b._val_folds
    assert splitter_a._test_folds == splitter_b._test_folds


def test_nacl_is_always_train_only(splitter_paths: dict[str, object]) -> None:
    """NaCl IDs must always stay in train and never appear in val/test."""
    splitter = _build_splitter(splitter_paths=splitter_paths, random_seed=5)

    all_group_ids = splitter_paths["all_group_ids"]
    nacl_group_ids = splitter_paths["nacl_group_ids"]

    for fold_index in range(splitter.num_folds):
        train_ids = set(splitter._train_folds[fold_index])
        val_ids = set(splitter._val_folds[fold_index])
        test_ids = set(splitter._test_folds[fold_index])

        assert nacl_group_ids.issubset(train_ids)
        assert nacl_group_ids.isdisjoint(val_ids)
        assert nacl_group_ids.isdisjoint(test_ids)

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert train_ids.union(val_ids).union(test_ids) == all_group_ids


def test_materialize_fold_and_validate_succeeds(splitter_paths: dict[str, object]) -> None:
    """Materialized folds should pass all consistency checks."""
    splitter = _build_splitter(splitter_paths=splitter_paths, random_seed=3)
    splitter.materialize_fold(clean_target_dir=True)

    splitter.validate_splits()


def test_validate_splits_detects_inconsistent_split(splitter_paths: dict[str, object]) -> None:
    """Validation should fail if one data branch is missing copied files."""
    splitter = _build_splitter(splitter_paths=splitter_paths, random_seed=7)
    splitter.materialize_fold(clean_target_dir=True)

    broken_file = next((splitter_paths["target_dir"] / "0" / "train" / "before" / "image").glob("*"))
    broken_file.unlink()

    with pytest.raises(ValueError, match="consistency check failed"):
        splitter.validate_splits()


def test_clean_target_path_removes_all_fold_directories(splitter_paths: dict[str, object]) -> None:
    """Cleanup should remove every generated fold directory."""
    splitter = _build_splitter(splitter_paths=splitter_paths, random_seed=0)
    splitter.materialize_fold(clean_target_dir=True)

    splitter.clean_target_path()

    for fold_index in range(splitter.num_folds):
        assert not (splitter.target_dir / f"{fold_index}").exists()


def test_raises_when_requested_fold_sizes_exceed_samples(splitter_paths: dict[str, object]) -> None:
    """Construction should fail when fold slicing asks for more IDs than available."""
    with pytest.raises(IndexError, match="Not enough samples"):
        _build_splitter(
            splitter_paths=splitter_paths,
            random_seed=0,
            num_folds=3,
            val_size=2,
            test_size=2,
        )
