import numpy as np
from typing import TypeVar, ClassVar, Callable, cast
from functools import wraps
from PIL.Image import open as png_open
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import zarr
from tifffile import imwrite
import logging


logger = logging.getLogger(__name__)

WORKERS = 12
IMAGE_SIZE = 512
SUPPORTED_FORMATS = (".png", ".bin")

Path_T = TypeVar("Path_T", list[Path | str], Path, str)
SPADData_T = TypeVar("SPADData_T", bound="SPADData")


def _try_load(func: Callable) -> Callable:
    """Decorator to catch errors when loading data.

    :param func: Method to load data.
    :type func: Callable
    :return: Decorated method
    :rtype: Callable
    """

    @wraps(func)
    def _error_check(path: Path | str, *args, **kwargs) -> np.ndarray:
        try:
            path = Path(path)
            output = func(path, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error @ try_load: {e}")
            output = _empty_array()
        return output

    return _error_check


@_try_load
def _bin(path: Path, *args, **kwarg) -> np.ndarray:
    """Loader of binary files.

    :param path: Path to the binary file.
    :type path: Path
    :return: 3D array of the data.
    :rtype: np.ndarray
    """
    with open(path, "rb") as f:
        raw_data = f.read()
    binframe_length = IMAGE_SIZE * IMAGE_SIZE // 8
    frame_count = len(raw_data) // binframe_length
    bits_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(
        frame_count, IMAGE_SIZE, IMAGE_SIZE // 8
    )
    return np.unpackbits(bits_array, axis=2)


@_try_load
def unbin(path: Path, data: np.ndarray | None = None) -> None:
    """Saves a 3D array back to a binary file.

    :param path: Path where the binary file will be saved.
    :type path: Path
    :param data: 3D array of the data to be saved. If None, a 3D numpy array of ones
        will be used.
    :type data: np.ndarray
    """
    if data is None:
        data = np.ones((100, 512, 512), dtype=np.uint8)
    packed_bits = np.packbits(data, axis=2)
    flat_data = packed_bits.ravel()
    with open(path, "wb") as f:
        f.write(flat_data.tobytes())


@_try_load
def _png(path: Path) -> np.ndarray:
    """Loader of png files using PIL.

    :param path: Path to the png file.
    :type path: Path
    :return: 3D array of the data.
    :rtype: np.ndarray
    """
    image_data = png_open(path)
    return np.array(image_data)[None, ...]


def _empty_array(*args, **kwargs) -> np.ndarray:
    """Factory of empty arrays.

    :return: The empty array.
    :rtype: np.ndarray
    """
    return np.array([])


class SPADData:
    """Class to handle SPAD data.

    :param path: Path to the file to be loaded. If None, the object will be created
        without data.
    :type path: Path | None
    """
    _default_loader: ClassVar[Callable] = staticmethod(_bin)

    def __init__(self, path: Path | None):
        """Class constructor."""
        self.path = path
        self.name = self.path.stem if self.path is not None else ""
        self.data = _empty_array()
        self.shape = self.data.shape
        self.z_sum = _empty_array()
        self.bin_size = 1
        self.z_bins = _empty_array()
        
    def __len__(self) -> int:
        self.load()
        return len(self.data)

    def __getitem__(self, index: int | slice) -> np.ndarray:
        return self.load()[index]

    def __str__(self) -> str:
        return str(self.name)
    
    def __repr__(self) -> str:
        return f"SPADData({self.path})"

    def load(self: SPADData_T, loader: Callable | None = None) -> SPADData_T:
        """Load data to the object.

        :param self: Self object.
        :type self: SPADData_T
        :param loader: Loader function to load the data from a file, defaults to None
        :type loader: Callable | None, optional
        :return: Object with the data loaded.
        :rtype: SPADData_T
        """
        if not self._loaded():
            try:
                load_data = self._default_loader if loader is None else loader
                self.data = load_data(self.path)
                self.shape = self.data.shape
            except Exception as e:
                logger.error(f"Error @ SPADData.load: {e}")
        return self

    def unload(self) -> None:
        """Remove data from the object to free memory."""
        if self._loaded():
            self.data = _empty_array()

    def _loaded(self) -> bool:
        return self.data.size > 0

    @classmethod
    def set_default_loader(cls, func: Callable) -> None:
        """Assign a loader method as the default loader.

        :param func: The loader method to be assigned.
        :type func: Callable
        """
        cls._default_loader = staticmethod(func)

    def __add__(self, other: "SPADData | np.ndarray") -> "SPADData":
        self.load()
        other_data = other.load().data if isinstance(other, SPADData) else other
        output = SPADData(path=None)
        output.data = np.concatenate([self.data, other_data], axis=0)
        output.name = (
            self.name + "+" + other.name
            if isinstance(other, SPADData)
            else self.name + "+others"
        )
        return output

    def __iadd__(self: SPADData_T, other: SPADData_T | np.ndarray) -> SPADData_T:
        self.load()
        self.path = None
        self.name = (
            self.name + "+" + other.name
            if isinstance(other, SPADData)
            else self.name + "+others"
        )
        other_data = other.load().data if isinstance(other, SPADData) else other
        self.data = np.concatenate([self.data, other_data], axis=0)
        return self

    def preview(self, plot: bool = False) -> np.ndarray:
        """Return a preview of the data.

        :param plot: Flag to plot the preview, defaults to False
        :type plot: bool, optional
        :return: The first frame of the data.
        :rtype: np.ndarray
        """
        self.load()
        if self.z_sum.size == 0:
            self.z_sum = np.sum(self.data, axis=0)
        if plot:
            import matplotlib.pyplot as plt

            plt.imshow(self.z_sum, cmap="gray")
            plt.title(f"Preview of {self.name}")
            plt.axis("off")
            plt.show()
        return self.z_sum

    def binz(self, bin_size: int = 10, unload_: bool = False) -> np.ndarray:
        """Bin the frames of the data.

        :param bin_size: Size of the bin, defaults to 10
        :type bin_size: int, optional
        :param unload_: Flag to unload the original data, defaults to False
        :type unload_: bool, optional
        :return: Binned data.
        :rtype: np.ndarray
        """
        self.load()
        self.bin_size = bin_size
        if self.data.shape[0] % bin_size != 0:
            logger.warning(
                f"Number of frames ({self.data.shape[0]}) is not divisible by bin size ({bin_size})."
            )
        try:
            self.z_bins = np.sum(
                self.data.reshape(-1, bin_size, *self.data.shape[1:]), axis=1
            )
        except Exception as e:
            logger.error(f"Error @ SPADData.bin_frames: {e}")
        if unload_:
            self.unload()
        return self.z_bins
    
    def __call__(self) -> np.ndarray:
        return self.load().data


class SPADFile:
    """The SPADFile class is a container for SPADData objects.
    This class can be used to load and unload data from the SPADData objects.

    :param files: List of files to be loaded. If a directory is given, all files
        in the directory will be loaded. If a single file is given, only that file.
        It can also be a list of files.
    :type files: Path_T
    :param load_data: Flag to load data when the object is created, defaults to False
    :type load_data: bool, optional
    """

    def __init__(self, files: Path_T, load_data: bool = False):
        """Class constructor."""
        if isinstance(files, list):
            file_list = [Path(f) for f in files if Path(f).is_file()]
        else:
            file_list = Path(files)
            if file_list.is_dir():
                file_list = list(file_list.iterdir())
            else:
                file_list = [file_list]
        file_list = [f for f in file_list if Path(f).is_file()]
        file_list.sort(key=lambda x: x.stem)
        self.directory = file_list[0].parent
        self.name = self.directory.stem
        self.files = list(
            SPADData(f) for f in file_list if Path(f).suffix in SUPPORTED_FORMATS
        )
        if load_data:
            self.load_data()

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(f"SPADFile({self.files})")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int | slice) -> SPADData | list[SPADData]:
        self.load_data(index)
        if isinstance(index, int):
            return cast(SPADData, self.files[index])
        else:
            return cast(list[SPADData], self.files[index])

    def _load_one(self, file: SPADData) -> SPADData:
        return file.load()

    def _unload_one(self, spad_data: SPADData) -> SPADData:
        spad_data.unload()
        return spad_data

    def load_data(self, index: int | slice = slice(None)) -> SPADData | list[SPADData]:
        """Load data to the specified index.

        :param index: Index of the data to be loaded, defaults to slice(None)
        :type index: int | slice, optional
        :return: Loaded data.
        :rtype: SPADData | list[SPADData]
        """
        try:
            if isinstance(index, int):
                return self._load_one(self.files[index])
            else:
                with ThreadPoolExecutor(max_workers=WORKERS) as exec:
                    exec.map(self._load_one, self.files[index])
                return self.files[index]
        except Exception as e:
            logger.error(f"Error @ SPADFile.load_data: {e}")
            raise

    def unload_all(self) -> None:
        """Unload all data from the object.

        :return: None
        :rtype: None
        """
        for spad_data in self.files:
            self._unload_one(spad_data)

    def combine(
        self, index: int | slice = slice(None), concat: bool = True
    ) -> np.ndarray:
        """Combine data from the specified index.

        :param index: Index of the data to be combined, defaults to slice(None)
        :type index: int | slice, optional
        :param concat: Flag to concatenate data, defaults to True. If False, the data will be
            stacked.
        :type concat: bool, optional
        :param default: Default value to return if there is an error, defaults to _empty_array()
        :type default: np.ndarray, optional
        :return: Combined data.
        :rtype: np.ndarray
        """
        try:
            data = self.load_data(index)
            if isinstance(data, SPADData):
                logger.warning("Only one file loaded. Returning the data directly.")
                return data.data
            else:
                combine_func = np.concatenate if concat else np.stack
                combined_data = combine_func([f.data for f in data], axis=0)
                return combined_data
        except Exception as e:
            logger.error(f"Error @ SPADFile.combine: {e}")
        return _empty_array()

    def save(
        self,
        path: Path | str,
        file_type: str = "zarr",
        index: int | slice = slice(None),
        concat: bool = True,
        **kwargs,
    ) -> None:
        """Save data to a file.

        :param path: Path to save the file.
        :type path: Path | str
        :param file_type: the type of the output file, defaults to "zarr"
        :type file_type: str, optional
        :param index: The index to be saved, defaults to slice(None)
        :type index: int | slice, optional
        :param concat: Concatenate or stack data before saving, defaults to True (concatenation)
        :type concat: bool, optional
        """
        path = self.directory if path is None else Path(path)
        path.mkdir(parents=True, exist_ok=True)
        combined_data = self.combine(index, concat)
        save_functions = {
            "zarr": self._as_zarr,
            "tiff": lambda p, d: imwrite(
                p, d
            ),  # Wrapping imwrite to match expected function signature
        }
        try:
            save_functions[file_type](
                path / f"{self.name}.{file_type}", combined_data, **kwargs
            )
        except Exception as e:
            logger.error(f"Error @ SPADFile.save: {e}")

    def _as_zarr(
        self,
        path: Path | str,
        data: np.ndarray,
        chunk_size: tuple[int, ...] | None = None,
    ) -> zarr.Array | zarr.Group:
        """Function to save data as zarr."""
        chunk_size = chunk_size if chunk_size else (10, *data.shape[1:])
        zarr_data = zarr.open(
            str(path), mode="w", shape=data.shape, chunks=chunk_size, dtype=data.dtype
        )
        zarr_data[:] = data
        zarr_data.attrs["name"] = self.name
        zarr_data.attrs["shape"] = data.shape
        return zarr_data

    def frame_count(self) -> int:
        """Return the number of frames in the data.

        :return: Number of frames.
        :rtype: int
        """
        return len(self.files[0])
