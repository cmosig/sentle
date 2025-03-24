from setuptools import find_packages, setup

VERSION = '2025.3.1'
DESCRIPTION = 'Sentinel-1 and Sentinel-2 scalable downloader.'
LONG_DESCRIPTION = 'Sentinel-1 & Sentinel-2 data cubes at large scale (bigger-than-memory) on any machine with integrated cloud detection, snow masking, harmonization, merging, and temporal composites.'

# Setting up
setup(
    name="sentle",
    version=VERSION,
    author="Clemens Mosig",
    author_email="clemens.mosig@uni-leipzig.de",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    include_package_data=True,
    package_data={"sentle": ["./data/sentinel2_grid_stripped_with_epsg.gpkg", "./data/cloudmodel.pt"]},
    install_requires=[
        "pystac-client>=0.7.7", "pystac>=1.10.1", "rasterio>=1.3.10",
        "affine>=2.4.0", "pandas>=2.2.2", "numpy>=1.26.4", "shapely>=2.0.4",
        "zarr>=2.18.1", "geopandas>=0.14.4", "planetary_computer>=1.0.0",
        "xarray>=2024.5.0", "numcodecs>=0.12.1", "scipy>=1.13.0",
        "torch>=2.3.0", "joblib>=1.4.2", "tqdm>=4.66.4"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ])
