from pathlib import Path
import os
from enum import StrEnum

import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.v2.functional import to_dtype
import torch


class ImageLoader(StrEnum):
    MATPLOTLIB = "matplotlib"
    TORCHVISION = "torchvision"


class ImageDataset:
    """Dataset generator for loading images, heights, masks and model predictions.

    Builds lists of files for the requested splits/timepoints/data types and
    provides simple loading and iteration helpers that return data arrays together
    with the sample identifier.

    Attributes:
        data_dir (pathlib.Path): Root directory containing split subdirectories (e.g. ``test``, ``train``).
        sample_ids (Sequence[str] | None): Optional list of sample stems to restrict the dataset to specific examples.
        split_type (Sequence[str]): Splits to include (e.g. ["test"]).
        model (Optional[str]): Model identifier used to locate prediction files under the ``predictions`` dir.
        file_lists (dict): Mapping from dataset keys (like ``"test/after/image"``) to ordered lists of `pathlib.Path` objects.
    """

    def __init__(
            self, data_dir, model=None, split_type=["test"],
            sample_ids=None, image_loader="matplotlib"):
        """Create an ImageDataset.

        Args:
            data_dir (str | pathlib.Path): Root directory containing split folders (e.g. ``test/after/image``).
            model (str | None): Model name used to look up prediction files under ``predictions/model-<model>-best/<split>``.
            split_type (list[str]): Split names to include (default: ["test"]).
            sample_ids (list[str] | None): If provided, dataset will be limited to these sample stems.
            image_loader (str | ImageLoader): Which backend to use for loading PNG images ("matplotlib" or "torchvision").
        """
        self.sample_ids = sample_ids
        self.data_dir = Path(data_dir)
        self.split_type = split_type
        self.model = model
        self.predictions_dir = Path("predictions")

        image_loader = ImageLoader(image_loader)
        if image_loader == ImageLoader.MATPLOTLIB:
            self.load_image = lambda path: plt.imread(path)
        elif image_loader == ImageLoader.TORCHVISION:
            import torchvision
            self.load_image = lambda path: to_dtype(
                torchvision.io.decode_image(path),
                torch.float32, scale=True).detach().numpy()
        else:
            raise ValueError(f"Unsupported image loader: {image_loader}")

        # Determine the data directories for the relevant data files (for each split, time, and data type), model predictions, and masks
        self.data_dirs = {}
        for split in self.split_type:
            for time in ["before", "after"]:
                for data_type in ["height", "image"]:
                    self.data_dirs[f"{split}/{time}/{data_type}"] = self.data_dir / split / time / data_type
            self.data_dirs[f"{split}/mask"] = self.data_dir / split / "mask"
            if model is not None:
                self.data_dirs[f"model/{split}"] = self.predictions_dir / \
                    f"model-{model}-best" / split

        # Generate the file list for each directory
        self.file_lists = {key: self._generate_file_lists(
            data_dir) for key, data_dir in self.data_dirs.items()}

    def __len__(self):
        return len(next(iter(self.file_lists.values())))

    def _generate_file_lists(self, data_dir):
        """Generate a list of file paths for the given data directory.

        If `sample_ids` was provided at construction time, this will return the
        filtered list for those IDs; otherwise it returns all files in the directory.

        Args:
            data_dir (pathlib.Path): Directory to scan for files.

        Returns:
            list[pathlib.Path]: Ordered list of file paths.
        """
        if self.sample_ids is not None:
            return self._get_ids_files(data_dir, self.sample_ids)
        else:
            return self._get_all_files(data_dir)

    def load_file(self, file_path):
        """Load a file and return its contents as a numpy array.

        Args:
            file_path (pathlib.Path): Path to the file to load. Supports ``.png`` and ``.npy``.

        Returns:
            numpy.ndarray: Array containing the loaded image or height data.
        """
        if file_path.suffix == ".png":
            return self.load_image(file_path)
        elif file_path.suffix == ".npy":
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")

    def _get_all_files(self, data_dir):
        """List all files in the given directory, sorted by name.

        Args:
            data_dir (pathlib.Path): Directory to scan.

        Returns:
            list[pathlib.Path]: Sorted list of file paths.
        """
        file_list = os.listdir(data_dir)
        file_list.sort()
        path_list = ([data_dir / file for file in file_list])
        return path_list

    def _get_ids_files(self, data_dir, image_ids):
        """Return files matching the provided image IDs.

        For every `image_id` this looks for a ``.npy`` file first and falls back to
        ``.png``. If neither exists for an ID a ``FileNotFoundError`` is raised.

        Args:
            data_dir (pathlib.Path): Directory to look for files in.
            image_ids (Sequence[str]): Sample stems to locate (without extension).

        Returns:
            list[pathlib.Path]: List of found file paths corresponding to `image_ids` in the same order.

        Raises:
            FileNotFoundError: If no file is found for a given image ID.
        """
        data_list = []
        for image_id in image_ids:
            file = data_dir / f"{image_id}"

            numpy_file = file.with_suffix(".npy")
            png_file = file.with_suffix(".png")
            if numpy_file.exists():
                data_list.append(numpy_file)
            elif png_file.exists():
                data_list.append(png_file)
            else:
                raise FileNotFoundError(f"Files for image ID {image_id} not found.")
        return data_list

    def iterate_files(self, identifiers):
        """Yield loaded data arrays for the requested file list identifiers.

        Args:
            identifiers (Sequence[str]): Keys into ``self.file_lists`` (for example
                ``"test/after/height"``, ``"model/test"``, ``"test/mask"``).

        Yields:
            list: List of loaded numpy arrays for each identifier in the same order,
                with the sample stem appended as the last element.
        """
        for files in zip(*(self.file_lists[identifier] for identifier in identifiers)):
            loaded_data = [(self.load_file(file)) for file in files]
            loaded_data.append(files[0].stem)
            yield loaded_data

    def __iter__(self):
        """Default iterator for the dataset.

        By default this iterates over the test split and yields ``[height, model_pred, mask, id]``.

        Note:
            If `model` was not provided at construction time the key ``"model/test"`` may be
            absent from ``self.file_lists`` and iteration will fail — construct the
            dataset with ``model`` when model predictions are required.
        """
        yield from self.iterate_files(["test/after/height", "model/test", "test/mask"])
