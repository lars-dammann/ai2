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
    """
    Dataset generator for loading data and corresponding masks.
    """

    def __init__(
            self, data_dir, model=None, split_type=["test"],
            sample_ids=None, image_loader="matplotlib"):
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

        # Determine the data directories for the rrelevatn data  files (for each split, time, and data type), model predictions, and masks
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
        """Generate a list of file paths for the given data directory, optionally filtering by sample IDs."""
        if self.sample_ids is not None:
            return self._get_ids_files(data_dir, self.sample_ids)
        else:
            return self._get_all_files(data_dir)

    def load_file(self, file_path):
        """Load a file based on its extension."""
        if file_path.suffix == ".png":
            return self.load_image(file_path)
        elif file_path.suffix == ".npy":
            return np.load(file_path)

    def _get_all_files(self, data_dir):
        """List all files in the given directory, sorted by name."""
        file_list = os.listdir(data_dir)
        file_list.sort()
        path_list = ([data_dir / file for file in file_list])
        return path_list

    def _get_ids_files(self, data_dir, image_ids):
        """List files in the given directory that match the provided image IDs, checking for both .npy and .png extensions."""
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
        """Iterate through the files for the given identifiers, loading the data and yielding it as a list."""
        for files in zip(*(self.file_lists[identifier] for identifier in identifiers)):
            loaded_data = [(self.load_file(file)) for file in files]
            loaded_data.append(files[0].stem)
            yield loaded_data

    def __iter__(self):
        """Default iterator that yields height, model prediction, and mask files and sample ids for the test split."""
        yield from self.iterate_files(["test/after/height", "model/test", "test/mask"])
