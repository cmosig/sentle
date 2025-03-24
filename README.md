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
    S1_assets=["vh_asc", "vh_desc", "vv_asc", "vv_desc"],
    S2_apply_snow_mask=True,
    S2_apply_cloud_mask=True,
    time_composite_freq="7d",
    num_workers=10,
)
```
This code downloads data for a 40km by 40km area with one year of both Sentinel-1 and Sentinel-2. Clouds and snow are detected and replaced with NaNs. Data is also averaged every 7 days. 

Everything is parallelized across 10 workers and each worker immediately saves its results to the specified path to a `zarr_store`. This ensures you can download larger-than-memory cubes.

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
lexcube.Cube3DWidget(da.load().sel(band="B02"), vmin=0, vmax=4000)
```

![image](https://github.com/user-attachments/assets/13c4688a-be9d-4a43-adac-63536756f5e9)

## API Documentation

### sentle.process

The package contains only one main function for retrieving and processing Sentinel data: `process`.

#### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_crs` | `rasterio.crs.CRS` | Specifies the target CRS that all data will be reprojected to. |
| `target_resolution` | `float` | Determines the resolution that all data is reprojected to in the `target_crs`. |
| `bound_left` | `float` | Left bound of area that is supposed to be covered. Unit is in `target_crs`. |
| `bound_bottom` | `float` | Bottom bound of area that is supposed to be covered. Unit is in `target_crs`. |
| `bound_right` | `float` | Right bound of area that is supposed to be covered. Unit is in `target_crs`. |
| `bound_top` | `float` | Top bound of area that is supposed to be covered. Unit is in `target_crs`. |
| `datetime` | `DatetimeLike` | Specifies time range of data to be downloaded. This is forwarded to the respective STAC interface. |
| `zarr_store` | `str` or `zarr.storage.Store` | Path of where to create the zarr storage. |

#### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `processing_spatial_chunk_size` | `int` | `4000` | Size of spatial chunks across which parallelization is performed. |
| `S1_assets` | `list[str]` | `["vh", "vv"]` | Specify which bands to download for Sentinel-1. Only "vh" and "vv" are supported. |
| `S2_mask_snow` | `bool` | `False` | Whether to create a snow mask. Based on https://doi.org/10.1016/j.rse.2011.10.028. |
| `S2_cloud_classification` | `bool` | `False` | Whether to create cloud classification layer, where `0=clear sky`, `2=thick cloud`, `3=thin cloud`, `4=shadow`. |
| `S2_cloud_classification_device` | `str` | `"cpu"` | On which device to run cloud classification. Either `"cpu"` or `"cuda"`. |
| `S2_return_cloud_probabilities` | `bool` | `False` | Whether to return raw cloud probabilities which were used to determine the cloud classes. |
| `num_workers` | `int` | `1` | Number of cores to scale computation across. Plan 2GiB of RAM per worker. -1 uses all available cores. |
| `time_composite_freq` | `str` | `None` | Rounding interval across which data is averaged. |
| `S2_apply_snow_mask` | `bool` | `False` | Whether to replace snow with NaN. |
| `S2_apply_cloud_mask` | `bool` | `False` | Whether to replace anything that is not clear sky with NaN. |
| `overwrite` | `bool` | `False` | Whether to overwrite existing zarr storage. |
| `zarr_store_chunk_size` | `dict` | `{"time": 50, "x": 100, "y": 100}` | Chunk sizes for zarr storage. |

#### Notes

- If `S2_apply_snow_mask` is set to `True`, `S2_mask_snow` must also be `True`.
- If `S2_apply_cloud_mask` is set to `True`, `S2_cloud_classification` must also be `True`.
- If `time_composite_freq` is set and neither `S2_apply_snow_mask` nor `S2_apply_cloud_mask` is set, a warning will be issued as temporal aggregation may yield useless results for Sentinel-2 data.
- When `S1_assets` is supplied as an empty list, it will be converted to `None`, meaning no Sentinel-1 data will be downloaded.

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
