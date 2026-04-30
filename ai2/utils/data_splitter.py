"""Utilities for deterministic cross-validation split generation."""

import os
import re
from pathlib import Path
import random
import shutil
from typing import Dict, List, Sequence

import pandas as pd


class CrossValidationDataSplitter:
    """Generate and materialize k-fold splits for corrosion data.

    This splitter groups files by sample base IDs (for example ``x-x-x`` from
    ``x-x-x-1.png``), enforces that NaCl samples are always in training, and
    supports fold-by-fold overwrite of one shared output directory.

    Args:
        original_path: Root directory containing ``before``, ``after``, and ``mask``.
        inhibitor_list_file: CSV metadata file with inhibitor names and sample ids.
        save_path: Destination directory where train/val/test splits are copied.
        num_folds: Number of validation folds to generate from non-NaCl samples.
        random_seed: Seed used for deterministic fold shuffling.
    """
    MASK_PATH = "mask"
    BEFORE_PATH = "before"
    AFTER_PATH = "after"
    IMAGE_PATH = "image"
    HEIGHT_PATH = "height"

    _SAMPLE_BASE_PATTERN = re.compile(r"^(?P<sample_group_id>.+)-\d+$")

    def __init__(
        self,
        source_dir: Path,
        inhibitor_list_file: Path,
        target_dir: Path,
        num_folds: int = None,
        random_seed: int = 0,
        val_size: int = 25,
        test_size: int = 25,
    ) -> None:

        self.source_dir = Path(source_dir)
        self.inhibitor_list_file = Path(inhibitor_list_file)
        self.target_dir = Path(target_dir)
        self.num_folds = num_folds
        self.random_seed = random_seed
        self.val_size = val_size
        self.test_size = test_size

        self._all_sample_group_ids = self._get_sample_group_ids_from_path(
            self.source_dir / self.MASK_PATH)
        self._nacl_sample_group_ids = self._extract_nacl_sample_ids()
        self._train_folds, self._val_folds, self._test_folds = self._generate_fold_splits()

    @classmethod
    def _get_sample_group_ids_from_path(cls, dir: Path) -> List[str]:
        """Collect sample group IDs from directory.

        Returns:
            Sorted sample group IDs discovered in the source data directory.
        """
        return sorted({cls._extract_sample_group_id(name)
                       for name in cls._get_sample_ids_from_path(dir)})

    @classmethod
    def _get_sample_ids_from_path(cls, dir: Path) -> List[str]:
        """Collect sample IDs from directory.

        Returns:
            Sorted sample group IDs discovered in the source data directory.
        """
        sample_ids = [path.stem for path in dir.glob("*") if path.is_file()]
        if not sample_ids:
            raise ValueError(f"No sample files found in {dir}")
        return sorted(sample_ids)

    @classmethod
    def _extract_sample_group_id(cls, file_name: str) -> str:
        """Extract the base sample ID from a sample filename.

        Args:
            file_path: Sample file path whose stem ends in a numeric suffix.

        Returns:
            Base sample ID without the trailing sample index.
        """
        match = cls._SAMPLE_BASE_PATTERN.match(file_name)
        if match is None:
            raise ValueError(f"Could not extract sample base ID from {file_name}")
        return match.group("sample_group_id")

    def _extract_nacl_sample_ids(self) -> List[str]:
        """Extract NaCl group IDs that are present in the discovered dataset.

        Returns:
            Sorted list of NaCl group sample IDs found in source files.
        """
        inhibitors = pd.read_csv(self.inhibitor_list_file)
        nacl_rows = inhibitors[inhibitors["name"].str.contains("NaCl", case=False)]
        nacl_ids = {
            self._SAMPLE_BASE_PATTERN.match(sample_id).group("sample_group_id")
            for sample_id in nacl_rows["id"].astype(str)
        }
        return sorted(nacl_ids.intersection(self._all_sample_group_ids))

    def _split_into_folds(self, sample_ids: Sequence[str]) -> List[List[str]]:
        """Split sample IDs into k folds.

        Args:
            sample_ids: Ordered list of sample IDs to distribute over folds.

        Returns:
            List of folds, each containing sample IDs.
        """
        train_folds = []
        val_folds = []
        test_folds = []

        # If no number of folds is specified, set it to the maximum possible given the val size
        if self.num_folds is None:
            self.num_folds = len(sample_ids) // self.val_size

        for fold in range(self.num_folds):
            train_sample_ids, val_sample_ids, test_sample_ids = self._extract_fold_sample_ids(
                fold, sample_ids)
            train_folds.append(sorted(train_sample_ids))
            val_folds.append(sorted(val_sample_ids))
            test_folds.append(sorted(test_sample_ids))

        return train_folds, val_folds, test_folds

    def _extract_fold_sample_ids(self, fold_index: int, sample_ids: Sequence[str]) -> List[str]:
        """Extract sample IDs for a specific fold.

        Args:
            fold_index: Zero-based fold index.
            sample_ids: List of sample IDs to distribute.

        Returns:
            List of sample IDs for the specified fold.
        """
        # Copy the original samples ids
        train_sample_ids = sample_ids.copy()

        # Determine the start and end indices for the val/test split for this fold
        val_test_start = self.val_size * fold_index
        val_test_end = val_test_start + (self.val_size + self.test_size)

        # Extract the val/test sample ids for this fold and remove them from the training sample ids
        val_test_samples_ids = train_sample_ids[val_test_start:val_test_end]
        if (val_test_start + self.val_size) > len(sample_ids):
            raise IndexError(
                f"Not enough samples ({len(sample_ids)}) to satisfy val split for fold "
                f"{fold_index + 1} with {self.val_size} validation samples (would require {(fold_index + 1) * self.val_size}).")
        val_sample_ids = val_test_samples_ids[:self.val_size]
        test_sample_ids = val_test_samples_ids[self.val_size:]

        # Delete the val/test sample ids from the training sample ids
        del train_sample_ids[val_test_start:val_test_end]

        return train_sample_ids, val_sample_ids, test_sample_ids

    def _filter_sample_ids(
            self, sample_ids: Sequence[str],
            exclude_ids: Sequence[str]) -> List[str]:
        """Filter out specific sample IDs from a list.

        Args:
            sample_ids: List of sample IDs to filter.
            exclude_ids: Sample IDs to exclude from the result.

        Returns:
            List of sample IDs after filtering.
        """
        return [sample_id for sample_id in sample_ids if sample_id not in exclude_ids]

    def _generate_fold_splits(self) -> List[Dict[str, List[str]]]:
        """Create train/val/test ID lists for each fold.

        Returns:
            List of split dictionaries with keys ``train``, ``val``, ``test``.
        """
        # Filter out all NaCl and problem sample IDs, shuffle the rest
        filtered_ids = self._filter_sample_ids(
            self._all_sample_group_ids, self._nacl_sample_group_ids)
        random.Random(self.random_seed).shuffle(filtered_ids)

        train_folds, val_folds, test_folds = self._split_into_folds(filtered_ids)

        # Add the filtered samples back to the training folds
        train_folds = [
            sorted((train_fold + self._nacl_sample_group_ids))
            for train_fold in train_folds]

        return train_folds, val_folds, test_folds

    def clean_target_path(self) -> None:
        """Delete existing split directories.

        Returns:
            None.
        """
        for fold in range(self.num_folds):
            split_dir = self.target_dir / f"{fold}"
            if split_dir.exists():
                shutil.rmtree(split_dir)

    def _copy_dataset(
            self, id_list: Sequence[str],
            fold_index: int, split_name: str, overwrite: bool = False) -> None:
        """Copy all files matching sample IDs into one split directory.

        Args:
            id_list: Sample base IDs to copy.
            fold_index: Zero-based fold index.
            split_name: Split directory name, one of train/val/test.
            overwrite: If True, overwrite existing files in the target directory.

        Returns:
            None.
        """
        for time_name in [self.BEFORE_PATH, self.AFTER_PATH]:
            for data_type in [self.IMAGE_PATH, self.HEIGHT_PATH]:
                source_dir = self.source_dir / time_name / data_type
                target_dir = self.target_dir / f"{fold_index}" / split_name / time_name / data_type
                self._copy_files(id_list, source_dir, target_dir, overwrite=overwrite)

        source_mask_dir = self.source_dir / "mask"
        target_mask_dir = self.target_dir / f"{fold_index}" / split_name / "mask"
        target_mask_dir.mkdir(parents=True, exist_ok=True)
        self._copy_files(id_list, source_mask_dir, target_mask_dir, overwrite=overwrite)

    def _copy_files(
            self, id_list: Sequence[str],
            source_dir: Path, target_dir: Path, overwrite: bool = False) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in id_list:
            for source_file in source_dir.rglob(f"{sample_id}-*"):
                if not overwrite and (target_dir / source_file.name).exists():
                    raise FileExistsError(
                        f"Target file {target_dir / source_file.name} already exists.")
                shutil.copy(source_file, target_dir)

    def materialize_fold(self, overwrite=False, clean_target_dir=False) -> Dict[str, List[str]]:
        """Overwrite target path with the selected fold's train/val/test files.

        Args:
            overwrite: If True, overwrite existing files in the target directory. If False, raise an error if target files already exist.
            clean_target_dir: If True, delete all existing fold directories before materializing the new fold.
        Returns:
            None.
        """
        if clean_target_dir:
            self.clean_target_path()

        for fold_index in range(self.num_folds):
            self._copy_dataset(
                self._train_folds[fold_index],
                fold_index, "train", overwrite=overwrite)
            self._copy_dataset(self._val_folds[fold_index], fold_index, "val", overwrite=overwrite)
            self._copy_dataset(
                self._test_folds[fold_index],
                fold_index, "test", overwrite=overwrite)

    def validate_splits(self) -> None:
        """Validate that the generated splits are mutually exclusive and complete."""

        for fold_index in range(self.num_folds):
            self._get_sample_ids_from_path

            # Validate that the before/height/image after/height/image and mask files contain the same sample IDs
            self._validate_split_consistency(self.target_dir / f"{fold_index}")

            # Validate that the train/val/test splits are mutually exclusive
            self._validate_exclusivity(self.target_dir / f"{fold_index}")

            # Validate that the train/val/test splits together contain all sample IDs
            self._validate_completeness(self.target_dir / f"{fold_index}")

    def _validate_split_consistency(self, fold_dir: Path) -> None:
        """Validate that all data types contain the same sample IDs within a split."""
        for split in ["train", "val", "test"]:
            split_dir = fold_dir / split

            compare_ids = set(self._get_sample_ids_from_path(split_dir / self.MASK_PATH))
            for time_name in [self.BEFORE_PATH, self.AFTER_PATH]:
                for data_type in [self.IMAGE_PATH, self.HEIGHT_PATH]:
                    dir = split_dir / time_name / data_type
                    if not dir.exists():
                        raise ValueError(
                            f"Expected directory {dir} does not exist for split consistency check.")

                    if not compare_ids == set(self._get_sample_ids_from_path(dir)):
                        raise ValueError(
                            f"Data type consistency check failed for {split_dir} between mask and {time_name}/{data_type}")

    def _validate_exclusivity(self, fold_dir: Path) -> None:
        """Validate that train/val/test splits are mutually exclusive."""
        train_ids = set(self._get_sample_ids_from_path(fold_dir / "train" / self.MASK_PATH))
        val_ids = set(self._get_sample_ids_from_path(fold_dir / "val" / self.MASK_PATH))
        test_ids = set(self._get_sample_ids_from_path(fold_dir / "test" / self.MASK_PATH))

        if train_ids.intersection(val_ids):
            raise ValueError(f"Train and val splits are not mutually exclusive in {fold_dir}")
        if train_ids.intersection(test_ids):
            raise ValueError(f"Train and test splits are not mutually exclusive in {fold_dir}")
        if val_ids.intersection(test_ids):
            raise ValueError(f"Val and test splits are not mutually exclusive in {fold_dir}")

    def _validate_completeness(self, fold_dir: Path) -> None:
        """Validate that train/val/test splits together contain all sample IDs."""
        train_ids = set(self._get_sample_ids_from_path(fold_dir / "train" / self.MASK_PATH))
        val_ids = set(self._get_sample_ids_from_path(fold_dir / "val" / self.MASK_PATH))
        test_ids = set(self._get_sample_ids_from_path(fold_dir / "test" / self.MASK_PATH))

        complete_sample_ids = set(self._get_sample_ids_from_path(self.source_dir / self.MASK_PATH))
        all_split_ids = train_ids.union(val_ids).union(test_ids)
        if all_split_ids != complete_sample_ids:
            raise ValueError(
                f"Train/val/test splits in {fold_dir} do not together contain all sample IDs")


if __name__ == "__main__":
    base_path = Path(os.path.dirname(__file__)).parent.parent / "crossval-data"
    target_path = base_path / "splits"
    source_path = base_path / "complete"
    inhibitor_list_file = source_path / "inhibitor-list.csv"

    splitter = CrossValidationDataSplitter(
        target_dir=target_path, source_dir=source_path,
        inhibitor_list_file=inhibitor_list_file, val_size=10, test_size=10, random_seed=1)
    splitter.materialize_fold(clean_target_dir=True)
    splitter.validate_splits()
    print(f"Generated {splitter.num_folds} folds with NaCl samples: "
          f"{len(splitter._nacl_sample_group_ids)}")
