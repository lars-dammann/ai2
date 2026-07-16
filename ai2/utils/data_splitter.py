"""Utilities for deterministic cross-validation split generation."""

import os
import re
from pathlib import Path
import random
import shutil
from typing import List, Sequence, Tuple

import pandas as pd


class CrossValidationDataSplitter:
    """Generate and materialize k-fold splits for corrosion data.

    This splitter groups files by sample base IDs (for example ``x-x-x`` from
    ``x-x-x-1.png``) and enforces that NaCl samples are kept in training.

    Args:
        source_dir (pathlib.Path): Root directory containing ``before``, ``after``, and ``mask`` folders.
        modulator_list_file (pathlib.Path): CSV metadata file with modulator names and sample ids.
        target_dir (pathlib.Path): Destination directory where train/val/test splits are copied.
        random_seed (int, optional): Seed used for deterministic fold shuffling. Defaults to 0.
        val_size (int, optional): Number of samples to use for validation per fold. Defaults to 25.
        test_size (int, optional): Number of samples to use for testing per fold. Defaults to 25.
    """
    MASK_PATH = "mask"
    BEFORE_PATH = "before"
    AFTER_PATH = "after"
    IMAGE_PATH = "image"
    HEIGHT_PATH = "height"

    # Regular expression pattern to extract the base sample ID from a sample filename.
    _SAMPLE_BASE_PATTERN = re.compile(r"^(?P<sample_group_id>.+)-\d+$")

    def __init__(
        self,
        source_dir: Path,
        modulator_list_file: Path,
        target_dir: Path,
        random_seed: int = 0,
        val_size: int = 25,
        test_size: int = 25,
    ) -> None:

        self.source_dir = Path(source_dir)
        self.modulator_list_file = Path(modulator_list_file)
        self.target_dir = Path(target_dir)
        self.num_folds = None
        self.random_seed = random_seed
        self.val_size = val_size
        self.test_size = test_size

        self._all_sample_group_ids = self._get_sample_group_ids_from_path(
            self.source_dir / self.MASK_PATH)
        self._nacl_sample_group_ids = self._extract_nacl_sample_ids()
        self._train_folds, self._val_folds, self._test_folds = self._generate_fold_splits()
        self.num_folds = len(self._test_folds)

    @classmethod
    def _get_sample_group_ids_from_path(cls, dir: Path) -> List[str]:
        """Collect sample group IDs from a directory.

        Args:
            dir (pathlib.Path): Directory containing sample files whose stems end with ``-<index>``.

        Returns:
            list[str]: Sorted unique sample group IDs (e.g. ``x-x-x``).
        """
        return sorted({cls._extract_sample_group_id(name)
                       for name in cls._get_sample_ids_from_path(dir)})

    @classmethod
    def _get_sample_ids_from_path(cls, dir: Path) -> List[str]:
        """List file stems from a directory.

        Args:
            dir (pathlib.Path): Directory to scan for files.

        Returns:
            list[str]: Sorted list of file stems (e.g. ``x-x-x-1``).

        Raises:
            ValueError: If the directory contains no files.
        """
        sample_ids = [path.stem for path in dir.glob("*") if path.is_file()]
        if not sample_ids:
            raise ValueError(f"No sample files found in {dir}")
        return sorted(sample_ids)

    @classmethod
    def _extract_sample_group_id(cls, file_name: str) -> str:
        """Extract the base sample ID from a file stem.

        Args:
            file_name (str): File stem expected to end with a numeric suffix (e.g. ``x-x-x-1``).

        Returns:
            str: Base sample ID without the trailing numeric index (e.g. ``x-x-x``).

        Raises:
            ValueError: If the file stem does not match the expected pattern.
        """
        match = cls._SAMPLE_BASE_PATTERN.match(file_name)
        if match is None:
            raise ValueError(f"Could not extract sample base ID from {file_name}")
        return match.group("sample_group_id")

    def _extract_nacl_sample_ids(self) -> List[str]:
        """Extract NaCl sample group IDs present in the dataset.

        The method reads the modulators CSV, selects rows whose ``name`` contains
        "NaCl" (case-insensitive), extracts their base sample IDs and returns the
        intersection with the sample IDs discovered on disk.

        Returns:
            list[str]: Sorted list of NaCl sample group IDs present in the source data.
        """
        modulators = pd.read_csv(self.modulator_list_file)
        nacl_rows = modulators[modulators["name"].str.contains("NaCl", case=False)]
        nacl_ids = {
            self._SAMPLE_BASE_PATTERN.match(sample_id).group("sample_group_id")
            for sample_id in nacl_rows["id"].astype(str)
        }
        return sorted(nacl_ids.intersection(self._all_sample_group_ids))

    def _split_into_folds(self, sample_ids: Sequence[str]) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """Split sample IDs into k folds.

        Args:
            sample_ids (Sequence[str]): Ordered list of sample IDs to distribute over folds.

        Returns:
            tuple[list[list[str]], list[list[str]], list[list[str]]]:
                (train_folds, val_folds, test_folds) where each element is a list of folds
                and each fold is a list of sample IDs.
        """
        train_folds = []
        val_folds = []
        test_folds = []

        # If no number of folds is specified, set it to the maximum possible given the val size
        num_folds = len(sample_ids) // self.val_size

        for fold in range(num_folds):
            train_sample_ids, val_sample_ids, test_sample_ids = self._extract_fold_sample_ids(
                fold, sample_ids, reverse=False)
            train_folds.append(sorted(train_sample_ids))
            val_folds.append(sorted(val_sample_ids))
            test_folds.append(sorted(test_sample_ids))

        # Include the last missing test set
        train_sample_ids, val_sample_ids, test_sample_ids = self._extract_fold_sample_ids(
            0, sample_ids, reverse=True)
        train_folds.append(sorted(train_sample_ids))
        val_folds.append(sorted(val_sample_ids))
        test_folds.append(sorted(test_sample_ids))

        return train_folds, val_folds, test_folds

    def _extract_fold_sample_ids(
            self, fold_index: int, sample_ids: Sequence[str],
            reverse: bool = False) -> Tuple[List[str], List[str], List[str]]:
        """Extract sample IDs for a specific fold.

        Args:
            fold_index (int): Zero-based fold index.
            sample_ids (Sequence[str]): List of sample IDs to distribute.
            reverse (bool): If True, reverse the split ordering for validation/test.

        Returns:
            tuple[list[str], list[str], list[str]]: (train_sample_ids, val_sample_ids, test_sample_ids).

        Raises:
            IndexError: If there are not enough samples to satisfy the requested validation slice.
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
        if reverse:
            val_sample_ids = val_test_samples_ids[self.test_size:]
            test_sample_ids = val_test_samples_ids[:self.test_size]
        else:
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
            sample_ids (Sequence[str]): List of sample IDs to filter.
            exclude_ids (Sequence[str]): Sample IDs to exclude from the result.

        Returns:
            list[str]: Filtered sample IDs.
        """
        return [sample_id for sample_id in sample_ids if sample_id not in exclude_ids]

    def _generate_fold_splits(self) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """Create train/val/test ID lists for each fold.

        Returns:
            tuple[list[list[str]], list[list[str]], list[list[str]]]:
                (train_folds, val_folds, test_folds) where each element is a list of folds
                and each fold is a list of sample group IDs.
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
        """Delete existing split directories under the target path.

        This removes any directory named by the fold index (``target_dir/0``, ``target_dir/1``, ...).

        Returns:
            None
        """
        for fold in range(self.num_folds):
            split_dir = self.target_dir / f"{fold}"
            if split_dir.exists():
                shutil.rmtree(split_dir)

    def _copy_dataset(
            self, id_list: Sequence[str],
            fold_index: int, split_name: str, overwrite: bool = False) -> None:
        """Copy files matching sample IDs into a split directory for a fold.

        Args:
            id_list (Sequence[str]): Sample base IDs to copy.
            fold_index (int): Zero-based fold index.
            split_name (str): Split directory name, one of 'train', 'val', 'test'.
            overwrite (bool): If True, overwrite existing files in the target directory.

        Returns:
            None
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
        """Copy files for the given sample IDs from source to target.

        Args:
            id_list (Sequence[str]): Sample base IDs to copy.
            source_dir (pathlib.Path): Directory containing source files to search.
            target_dir (pathlib.Path): Destination directory where files will be copied.
            overwrite (bool): If True, overwrite existing files in the target directory.

        Raises:
            FileExistsError: If a target file already exists and ``overwrite`` is False.

        Returns:
            None
        """

        target_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in id_list:
            for source_file in source_dir.rglob(f"{sample_id}-*"):
                if not overwrite and (target_dir / source_file.name).exists():
                    raise FileExistsError(
                        f"Target file {target_dir / source_file.name} already exists.")
                shutil.copy(source_file, target_dir)

    def materialize_fold(self, overwrite: bool = False, clean_target_dir: bool = False) -> None:
        """Materialize all generated folds under the target directory.

        Args:
            overwrite (bool): If True, overwrite existing files in the target directory. If False, raise an error if target files already exist.
            clean_target_dir (bool): If True, delete all existing fold directories before materializing the new fold.

        Returns:
            None
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
        """Validate all generated folds.

        For each generated fold this runs consistency, exclusivity and completeness
        checks. Any failing check raises a ``ValueError`` with details.

        Raises:
            ValueError: If any per-fold validation check fails.

        Returns:
            None
        """

        for fold_index in range(self.num_folds):
            # Validate that the before/height/image after/height/image and mask files contain the same sample IDs
            self._validate_split_consistency(self.target_dir / f"{fold_index}")

            # Validate that the train/val/test splits are mutually exclusive
            self._validate_exclusivity(self.target_dir / f"{fold_index}")

            # Validate that the train/val/test splits together contain all sample IDs
            self._validate_completeness(self.target_dir / f"{fold_index}")

    def _validate_split_consistency(self, fold_dir: Path) -> None:
        """Validate that all data types contain the same sample IDs within a split.

        For each split ("train", "val", "test") this verifies that the set of sample
        IDs present in the mask directory matches the set present in each of the
        corresponding before/after image/height directories.

        Args:
            fold_dir (pathlib.Path): Path to the fold directory to validate (e.g. target_dir/0).

        Raises:
            ValueError: If an expected subdirectory is missing or the sample ID sets do not match.

        Returns:
            None
        """

        for split in ["train", "val", "test"]:
            split_dir = fold_dir / split

            compare_ids = set(self._get_sample_ids_from_path(split_dir / self.MASK_PATH))
            for time_name in [self.BEFORE_PATH, self.AFTER_PATH]:
                for data_type in [self.IMAGE_PATH, self.HEIGHT_PATH]:
                    data_dir = split_dir / time_name / data_type
                    if not data_dir.exists():
                        raise ValueError(
                            f"Expected directory {data_dir} does not exist for split consistency check.")

                    if compare_ids != set(self._get_sample_ids_from_path(data_dir)):
                        raise ValueError(
                            f"Data type consistency check failed for {split_dir} between mask and {time_name}/{data_type}")

    def _validate_exclusivity(self, fold_dir: Path) -> None:
        """Validate that train/val/test splits are mutually exclusive.

        Args:
            fold_dir (pathlib.Path): Path to the fold directory to validate (e.g. target_dir/0).

        Raises:
            ValueError: If any pair of splits share sample IDs.

        Returns:
            None
        """

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
        """Validate that train/val/test splits together contain all sample IDs.

        Args:
            fold_dir (pathlib.Path): Path to the fold directory to validate (e.g. target_dir/0).

        Raises:
            ValueError: If the union of train/val/test sample IDs does not equal the complete
                set of sample IDs discovered in the source data mask directory.

        Returns:
            None
        """

        train_ids = set(self._get_sample_ids_from_path(fold_dir / "train" / self.MASK_PATH))
        val_ids = set(self._get_sample_ids_from_path(fold_dir / "val" / self.MASK_PATH))
        test_ids = set(self._get_sample_ids_from_path(fold_dir / "test" / self.MASK_PATH))

        complete_sample_ids = set(self._get_sample_ids_from_path(self.source_dir / self.MASK_PATH))
        all_split_ids = train_ids.union(val_ids).union(test_ids)
        if all_split_ids != complete_sample_ids:
            raise ValueError(
                f"Train/val/test splits in {fold_dir} do not together contain all sample IDs")
