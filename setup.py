from setuptools import setup, find_packages

VERSION = '2024.5.0'
DESCRIPTION = 'Sentinel-2 scalable downloader.'
LONG_DESCRIPTION = 'Download Sentinel-2 data cubes of any scale (larger-than-memory) on any machine with integrated cloud detection, snow masking, harmonization, merging, and temporal composites.'

# Setting up
setup(name="sentle",
      version=VERSION,
      author="Clemens Mosig",
      author_email="clemens.mosig@uni-leipzig.de",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      packages=find_packages(),
      install_requires=[
          "dask>=2024.5.0", "pystac-client>=0.7.7", "pystac>=1.10.1",
          "rasterio>=1.3.10", "affine>=2.4.0", "pandas>=2.2.2",
          "numpy>=1.26.4", "shapely>=2.0.4", "zarr>=2.18.1",
          "geopandas>=0.14.4", "planetary_computer>=1.0.0", "xarray>=2024.5.0",
          "distributed>=2024.5.0", "numcodecs>=0.12.1", "scipy>=1.13.0",
          "torch>=2.3.0"
      ],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Education",
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 3",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: Microsoft :: Windows",
      ])
