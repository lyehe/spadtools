Introduction
============

``spadtools`` is a Python package for working with SPAD files. 
SPADTools is a simple data loader and preprocessor for the SPAD data from the SPAD512S Photon-Counting SPAD Camera. It can load the entire SPAD data folder with optimized loading and unpacking speed (at least 10x faster than the vendor python loader). It support lazy loading, indexing and slicing of the data. It also support dummy data generation and statistical off-board hotpixel correction.
