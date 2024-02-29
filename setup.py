from setuptools import setup, find_packages

setup(
    name="spadtools",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python-headless",
        "zarr",
        "tifffile",
        "pillow",
        "scipy",
        "matplotlib",
    ]
)