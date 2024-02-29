import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from .spadio import unbin, SPADData
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
MAX_WORKER = 10

class SPADHotpixelTool:
    """Class to facilitate the hotpixel correction of SPAD data.

    :param data: SPAD data or 3D np array
    :type data: SPADData | np.ndarray
    :param kwargs: Additional parameters
    :type kwargs: dict
    """

    def __init__(self, data: SPADData | np.ndarray, **kwargs):
        # Hint: you can preload the hotpixel locations from a different source
        self.data = data if isinstance(data, np.ndarray) else data.data
        self.flattend_data = self._flatten(self.data)
        if (zero_count := self._zero_count()) > 512:
            logger.warning(
                f"Possible insufficient data for correction. Zero count: {zero_count}"
            )
        self.reset()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self, *args: str) -> "SPADHotpixelTool":
        """Reset the intermediate results of the hotpixel correction.

        :param args: Attributes to reset
        :type args: list[str]
        :return: The instance of the class
        :rtype: SPADHotpixelTool
        """
        if not args:
            self.background = None
            self.hotpixel_image = None
            self.hotpixel_locations = None
            self.deadpixel_locations = None
            self.hotpixel_values = None
            self.expected_values = None
            self.probability_list = None
            self.corrected_image = None
        for arg in args:
            try:
                setattr(self, arg, None)
            except AttributeError:
                logger.error(f"Attribute {arg} not found")
        return self

    def _zero_count(self) -> int:
        """Return the number of zero in the flattened data."""
        count = np.sum(self.flattend_data < 1).item()
        return count

    @staticmethod
    def _flatten(data: np.ndarray) -> np.ndarray:
        """Reduce the dimension of the input data to 2D and save it as a np array."""
        if data.ndim > 2:
            data = np.sum(data, axis=tuple(range(data.ndim - 2)))
        return data.astype(np.float32)

    def get_background(self, method: str = "open", kernel_size: int = 3) -> np.ndarray:
        """Apply background filter to the input data and return the last result.

        :param method: Method for background filter, defaults to "open"
        :type method: str, optional
        :return: Data after background filter
        :rtype: np.ndarray
        """
        image = self.flattend_data
        methods = {
            "open": lambda i, k: cv2.morphologyEx(
                i,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            ),
            "mean": lambda i, k: cv2.blur(i, (k, k)),
            "median": lambda i, k: cv2.medianBlur(i.astype(np.float32), k),
            "median+mean": lambda i, k: methods["median"](methods["mean"](i, k), k),
        }
        if method in methods:
            self.background = methods[method](image, kernel_size)
        else:
            logger.error(f"Method {method} not implemented")
            raise NotImplementedError
        return self.background

    def get_hotpixels(self, **kwargs) -> np.ndarray:
        """Subtract background from the input data to get hot pixels.

        :return: Data after hot pixel filter
        :rtype: np.ndarray
        """
        if self.hotpixel_image is None:
            kernel_size: int = kwargs.get("kernel_size", 3)
            background_image = self.get_background(kernel_size=kernel_size)
            self.hotpixel_image = self.flattend_data - background_image
        return self.hotpixel_image

    def locate_hotpixels(
        self,
        hp_threshold: float = 2,
        **kwargs,
    ) -> np.ndarray:
        """Find hot pixels in the input data based on the standard deviation threshold.

        :param hp_threshold: Multiplyer for the standard deviation, defaults to 2
        :type hp_threshold: float, optional
        :return: The locations of the hot pixels
        :rtype: np.ndarray
        """
        if self.hotpixel_locations is None:
            hp_threshold = kwargs.get("hp_threshold", hp_threshold)
            hotpixel_image = self.get_hotpixels(**kwargs)
            mean = np.mean(hotpixel_image).item()
            std = np.std(hotpixel_image).item()
            hp_threshold = mean + hp_threshold * std
            self.hotpixel_locations = np.argwhere(hotpixel_image > hp_threshold)
        return self.hotpixel_locations

    def locate_deadpixels(self, **kwargs) -> np.ndarray:
        """Find dead pixels in the input data based on the standard deviation dp_threshold.

        :param dp_threshold: Multiplyer for the standard deviation, defaults to 1
        :type dp_threshold: float, optional
        :return: The locations of the dead pixels
        :rtype: np.ndarray
        """
        if self.deadpixel_locations is None:
            dp_threshold = kwargs.get("dp_threshold", 1)
            self.deadpixel_locations = np.argwhere(self.flattend_data < dp_threshold)
        return self.deadpixel_locations

    def correct_deadpixels(self, **kwargs) -> np.ndarray:
        """Correct the input data for dead pixels.

        :return: The corrected image
        :rtype: np.ndarray
        """
        image = (
            np.array(self.data)
            if self.corrected_image is None
            else self.corrected_image
        )
        index = self.locate_deadpixels(**kwargs)
        kernel_size: int = kwargs.get("kernel_size", 3)
        background_image = self.get_background(kernel_size=kernel_size, method="median")
        expected_values = background_image[index[:, 0], index[:, 1]]
        for (x, y), value in zip(index, expected_values):
            prob = value / image.shape[0]
            image[:, x, y] = np.random.binomial(1, prob, image.shape[0])
        self.corrected_image = image
        return self.corrected_image

    def get_hotpixel_values(self, **kwargs) -> np.ndarray:
        """Get the values of the hot pixels in the input data.

        :return: The values of the hot pixels
        :rtype: np.ndarray
        """
        if self.hotpixel_values is None:
            image = self.flattend_data
            index = self.locate_hotpixels(**kwargs)
            self.hotpixel_values = image[index[:, 0], index[:, 1]]
        return self.hotpixel_values

    def get_expected_values(self, **kwargs) -> np.ndarray:
        """Get the values of the hot pixels in the background image.

        :return: The values of the hot pixels
        :rtype: np.ndarray
        """
        if self.expected_values is None:
            kernel_size: int = kwargs.get("kernel_size", 3)
            background_image = self.get_background(kernel_size=kernel_size)
            index = self.locate_hotpixels(**kwargs)
            self.expected_values = background_image[index[:, 0], index[:, 1]]
        return self.expected_values

    def get_probablity(self, **kwargs) -> np.ndarray:
        """Get the probability of a hot pixel.

        :param probablity_base: Probability base, defaults to "hotpixels"
        :type probablity_base: str, optional
        :return: The probability of a hot pixel
        :rtype: np.ndarray
        """
        if self.probability_list is None:
            probablity_base = kwargs.get("probablity_base", "hotpixels")
            expected_values = self.get_expected_values(**kwargs)
            if probablity_base == "all":
                self.probability_list = expected_values / self.data.shape[0]
            elif probablity_base == "hotpixels":
                hotpixel_values = self.get_hotpixel_values(**kwargs)
                self.probability_list = expected_values / hotpixel_values
            else:
                logger.error(f"Probability base {probablity_base} not implemented")
                raise NotImplementedError
        return self.probability_list

    def correct_hotpixels(self, **kwargs) -> np.ndarray:
        """Correct the input data for hot pixels.

        :param probablity_base: Probability base, defaults to "hotpixels"
        :type probablity_base: str, optional
        :return: The corrected image
        :rtype: np.ndarray
        """
        probablity_base: str = kwargs.get("probablity_base", "hotpixels")
        image = (
            np.array(self.data)
            if self.corrected_image is None
            else self.corrected_image
        )
        index = self.locate_hotpixels(**kwargs)
        p = self.get_probablity(**kwargs)
        for (x, y), prob in zip(index, p):
            if probablity_base == "all":
                filling = np.random.binomial(1, prob, image.shape[0])
                image[:, x, y] = filling
            elif probablity_base == "hotpixels":
                z_idx = np.argwhere(image[:, x, y] == 1).flatten()
                image[z_idx, x, y] = np.random.binomial(1, prob, z_idx.size)
            else:
                logger.error(f"Probability base {probablity_base} not implemented")
        index = self.locate_deadpixels()
        self.corrected_image = image
        return self.corrected_image

    def inspect(self, **kwargs):
        """Show image results of the hotpixel correction."""
        try:
            import matplotlib.pyplot as plt

            deadpixel = kwargs.get("deadpixel", False)
            if deadpixel:
                self.correct_deadpixels(**kwargs)
            _, ax = plt.subplots(1, 2, figsize=(20, 40))
            ax[0].imshow(self.flattend_data, cmap="gray")
            ax[0].scatter(
                self.locate_hotpixels(**kwargs)[:, 1] + 3,
                self.locate_hotpixels(**kwargs)[:, 0],
                c="r",
                s=5,
                marker="_",
            )
            if deadpixel:
                ax[0].scatter(
                    self.locate_deadpixels(**kwargs)[:, 1] + 3,
                    self.locate_deadpixels(**kwargs)[:, 0],
                    c="g",
                    s=5,
                    marker="_",
                )
            title = f"Hotpixels (count: {self.locate_hotpixels(**kwargs).shape[0]})"
            if deadpixel:
                title += f" & Deadpixels (count: {self.locate_deadpixels(**kwargs).shape[0]})"
            ax[0].set_title(title)
            ax[0].set_axis_off()
            ax[1].imshow(np.sum(self.correct_hotpixels(**kwargs), axis=0), cmap="gray")
            ax[1].scatter(
                self.locate_hotpixels(**kwargs)[:, 1] + 3,
                self.locate_hotpixels(**kwargs)[:, 0],
                c="r",
                s=5,
                alpha=0.5,
                marker="_",
            )
            if deadpixel:
                ax[1].scatter(
                    self.locate_deadpixels(**kwargs)[:, 1] + 3,
                    self.locate_deadpixels(**kwargs)[:, 0],
                    c="g",
                    s=5,
                    alpha=0.5,
                    marker="_",
                )
            ax[1].set_title("Projection without hotpixels")
            ax[1].set_axis_off()
            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.show()
        except ImportError:
            logger.error("Matplotlib not installed")


class GenerateTestData:
    """Class to generate test data for the SPADFile class. Test data saved in binary format.

    :param output_path: Path to the output, defaults to Path("./example_data")
    :type output_path: Path, optional
    :param data: Data, defaults to binomial random data (p=0.05)
    :type data: np.ndarray, optional
    :param file_name: File name prefix, defaults to "test_data"
    :type file_name: str, optional
    :param number_of_files: Number of the generated test data, defaults to 20
    :type number_of_files: int, optional
    :return: Path to the generated test data.
    :rtype: Path
    """

    def __init__(
        self,
        output_path: Path = Path("./example_data"),
        p: float = 0.05,
        z_size: int = 100,
        file_name: str = "test_data",
        number_of_files: int = 20,
        number_of_hotpixels: int = 1000,
    ):
        """Constructor method."""
        self.output_path = output_path
        self.p = p
        self.z_size = z_size
        self.file_name = file_name + "_"
        self.number_of_files = number_of_files
        self.number_of_hotpixels = number_of_hotpixels

    def generate(self, *args) -> np.ndarray:
        """Generate test data using binomial sampling.

        :return: The generated test data
        :rtype: np.ndarray
        """
        data = np.random.binomial(1, self.p, (self.z_size, 512, 512))
        for _ in range(self.number_of_hotpixels):
            x = np.random.randint(0, 512)
            y = np.random.randint(0, 512)
            data[:, x, y] = np.random.binomial(
                1, 1 - np.random.random() / 2, self.z_size
            )

        return np.array(data)

    def create(self) -> Path:
        """Generate test data and save it to the specified path.

        :return: Path to the generated test data
        :rtype: Path
        """
        self.output_path.mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKER) as e:
            dummy_data = list(e.map(self.generate, range(self.number_of_files)))
            for i, data in enumerate(dummy_data):
                e.submit(
                    unbin,
                    self.output_path / f"{self.file_name}{i:0>4}.bin",
                    data
                )
        return self.output_path

    def remove(self):
        """Remove test data from the specified path."""
        try:
            for file in self.output_path.iterdir():
                file.unlink()
            self.output_path.rmdir()
        except FileNotFoundError:
            pass
