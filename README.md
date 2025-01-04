<p align="center">
  <a href="https://github.com/cmosig/sentle/"><img src="https://github.com/cmosig/sentle/raw/main/docs/logo.png" alt="sentle"></a>
</p>

<p align="center">
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://peps.python.org/pep-0008/" target="_blank">
    <img src="https://img.shields.io/badge/code_style-pep8-blue" alt="Black">
</a>
<a href="https://doi.org/10.5281/zenodo.13997085"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13997085.svg" alt="DOI"></a>

</p>
<p align="center">
    <em>Download Sentinel-1 & Sentinel-2 data cubes of huge-scale (larger-than-memory) on any machine with integrated cloud
detection, snow masking, harmonization, merging, and temporal composites.</em>
</p>

---
 
## Important Notes

1) **This package is in early alpha stage. There will be bugs!** If you encounter any error, warning, memory issue, etc. please open a GitHub issue with the code to reproduce.
2) This package is meant for large-scale processing and any area that is smaller than 8km in width and height will not run faster because of the underlying processing scheme. 

## Installing

This package is tested with Python 3.12.*. It may or may not work with other versions. 

```
pip install sentle
```
or 
```
git clone git@github.com:cmosig/sentle.git
cd sentle
pip install -e .
```

## Guide

**Process**

There is only one important function: `process`. Here, you specify all parameters necessary for download and processing. Once this function is called, it immediately starts downloading and processing the data you specified into a zarr file.

```
from sentle import sentle
from rasterio.crs import CRS

sentle.process(
    zarr_store="mycube.zarr",
    target_crs=CRS.from_string("EPSG:32633"),
    bound_left=176000,
    bound_bottom=5660000,
    bound_right=216000,
    bound_top=5700000,
    datetime="2022-06-17/2023-06-17",
    target_resolution=10,
    S2_mask_snow=True,
    S2_cloud_classification=True,
    S2_cloud_classification_device="cuda",
    S1_assets=["vv", "vh"],
    S2_apply_snow_mask=True,
    S2_apply_cloud_mask=True,
    time_composite_freq="7d",
    num_workers=10,
)
```
This code downloads data for a 40km by 40km area with one year of both Sentinel-1 and Sentinel-2. Clouds and snow are detected and replaced with NaNs. Data is also averaged every 7 days. 

Everything is parallelized across 10 workers and each worker immediately saves its results to the specified path to a `zarr_store`. This ensures you can download larger-than-memory cubes.

Explanation:
- `zarr_store`: Save path. 
- `target_crs`: Specifies the target CRS that all data will be reprojected to.
- `target_resolution`:  Determines the spatial resolution that all data is reprojected to in the `target_crs`. 
- `bound_*`: Spatial bounds in `target_crs` of the area you want to download. Undefined behavior if difference between opposite bounds is not divisible by `target_resolution`.
- `datetime`: Time range that will be downloaded.
- `S2_mask_snow`: Whether to compute snow mask for Sentinel-2 data.
- `S2_cloud_classification`: Whether to perform a cloud classification layer for Sentinel-2 data.
- `S2_cloud_classification_device`: Where to run cloud classification. If you have an Nvidia GPU then pass `cuda` otherwise `cpu`(default).
- `S2_apply_*`: Whether to apply the respective mask, i.e., replace values by NaN.
- `S1_assets`: Which Sentinel-1 assets to download. Disable Sentinel-1 by setting this to `None`.
- `time_composite_freq`: Rounding interval across which data is averaged. Uses `pandas.Timestamp.round(time_composite_freq)`. Cloud/snow masks are dropped after masking because they cannot be aggregated.
- `num_workers`: Number of cores to use. Plan about 2 GiB of memory usage per worker. -1 means all cores.
- `processing_spatial_chunk_size`: Size of spatial chunks that are processed in parallel. Default is 4000.
- `overwrite`: Whether to overwrite the zarr store if it already exists.  Default is False.

**Visualize**

Load the data with xarray. 

```
import xarray as xr
da = xr.open_zarr("mycube.zarr").sentle
da
```

<p align="center">
<img src="https://github.com/cmosig/sentle/assets/32590522/f487bba1-3c10-42a2-9b10-356ab2b44825" width="600">
</p>

And visualize using the awesome [lexcube](https://github.com/msoechting/lexcube) package. Here, band B02 is visualized from the above example. One is able to spot the cloud gaps and the spotty coverage during winter.

```
import lexcube
lexcube.Cube3DWidget(da.sel(band="B02"), vmin=0, vmax=4000)
```

![image](https://github.com/user-attachments/assets/13c4688a-be9d-4a43-adac-63536756f5e9)


## Questions you may have

#### How do I scale this program?
Increase the number of workers using the `num_workers` parameter when calling `sentle.process`. With default spatial chunk size of 4000, specified by `processing_spatial_chunk_size`, you should plan with 2GiB per worker.

## Contributing

Please submit issues or pull requests if you feel like something is missing or
needs to be fixed. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thank you to [Cesar Aybar](https://csaybar.github.io/) for his cloud detection model. All cloud detection in this package is performed using his model. The paper: [link](https://www.nature.com/articles/s41597-022-01878-2)
- Thank you to [David Montero](https://github.com/davemlz) for all the
discussions and his awesome packages which inspired this.
