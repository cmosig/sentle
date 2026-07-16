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
<a href="https://github.com/sacridini/Awesome-Geospatial" target="_blank"><img src="https://awesome.re/mentioned-badge.svg" alt="Mentioned in Awesome Geospatial"></a>

</p>
<p align="center">
    <em>Download Sentinel-1 & Sentinel-2 data cubes of huge-scale (larger-than-memory) on any machine with integrated cloud
detection, snow masking, harmonization, merging, and temporal composites.</em>
</p>

---

## Installing

This package is tested against Python 3.12 and 3.13 (see the `tests` workflow).
Older versions are not supported: sentle's current dependencies (rasterio,
numpy, scipy) require Python >=3.12.

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

sentle.process(
    zarr_store="mycube.zarr",
    target_crs="EPSG:32633",
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
    S2_nbar=True,
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

| Parameter           | Type                          | Description                                                                                                                                           |
| ------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `target_crs`        | `rasterio.crs.CRS` or `str`   | Specifies the target CRS that all data will be reprojected to. You can provide either a `rasterio.crs.CRS` object or a string (e.g., `"EPSG:32633"`). Both projected (e.g. UTM) and geographic (e.g. `"EPSG:4326"`) CRSs are supported. |
| `target_resolution` | `float`                       | Determines the resolution that all data is reprojected to, in the units of `target_crs`. May be fractional — e.g. `0.001` degrees for a lat/lon `EPSG:4326` cube, or `10` metres for a UTM cube.                                       |
| `bound_left`        | `float`                       | Left bound of area that is supposed to be covered. Unit is in `target_crs`.                                                                           |
| `bound_bottom`      | `float`                       | Bottom bound of area that is supposed to be covered. Unit is in `target_crs`.                                                                         |
| `bound_right`       | `float`                       | Right bound of area that is supposed to be covered. Unit is in `target_crs`.                                                                          |
| `bound_top`         | `float`                       | Top bound of area that is supposed to be covered. Unit is in `target_crs`.                                                                            |
| `datetime`          | `DatetimeLike`                | Specifies time range of data to be downloaded. This is forwarded to the respective STAC interface.                                                    |
| `zarr_store`        | `str` or `zarr.storage.Store` | Path of where to create the zarr storage.                                                                                                             |

#### Optional Parameters

| Parameter                        | Type                        | Default                                      | Description                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------- | --------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `provider`                       | `str`                       | `"planetary_computer"`                       | Which data catalog to download from: `"planetary_computer"` (default) or `"cdse"` (Copernicus Data Space Ecosystem). Both serve the same ESA L2A product, so reflectances are identical. CDSE is **Sentinel-2 only** (no Sentinel-1 RTC, no NBAR yet) and reads JP2s from CDSE S3, so it needs CDSE S3 credentials via the standard AWS chain (e.g. `AWS_PROFILE=cdse`, or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`). CDSE is slower than PC (JP2-over-S3 tile-structure discovery). |
| `reuse_open_datasets`            | `bool`                      | `True`                                       | Keep each Sentinel-2 band raster open and reuse it across all subtiles of the same tile within a spatial chunk (instead of re-opening per subtile), which also keeps GDAL's decoded-tile block cache warm. Mainly speeds up CDSE on the **older archive** (processing baseline < 05.12), where the first windowed read of a JP2 pays a one-time tile-structure discovery (~7 s cold); reuse amortizes it (≈2× faster on a 10 km area, more on larger ones). Newer CDSE products (baseline ≥ 05.12, from 2026) carry native TLM markers and are already fast per crop (~0.7 s cold) without it. Set to `False` for lower memory. |
| `processing_spatial_chunk_size`  | `int`                       | `4000`                                       | Size of spatial chunks across which parallelization is performed in pixels.                                                                                                                                                                                                                                                                               |
| `S1_assets`                      | `list[str]`                 | `["vh_asc", "vh_desc", "vv_asc", "vv_desc"]` | Specify which bands to download for Sentinel-1. Only "vh_asc", "vh_desc", "vv_asc", "vv_desc" are supported. Empty list will be converted to None (no Sentinel-1 data).                                                                                                                                                                                   |
| `S2_bands`                       | `list[str]`                 | all 12 raw bands                             | Which Sentinel-2 bands to download/save. Behaves like `S1_assets`: the default is all 12 raw bands. A subset list (e.g. `["B04", "B03", "B02"]` for RGB) downloads only those bands (not allowed together with `S2_cloud_classification`, which needs all bands; must include the bands `S2_mask_snow`/`S2_nbar` depend on). `None` or `[]` disables Sentinel-2 entirely for a Sentinel-1-only cube (requires `S1_assets`). Output band order always follows the raw-band order. |
| `S2_mask_snow`                   | `bool`                      | `False`                                      | Whether to create a snow mask. Based on https://doi.org/10.1016/j.rse.2011.10.028.                                                                                                                                                                                                                                                                        |
| `S2_cloud_classification`        | `bool`                      | `False`                                      | Whether to create cloud classification layer, where `0=clear sky`, `2=thick cloud`, `3=thin cloud`, `4=shadow`.                                                                                                                                                                                                                                           |
| `S2_cloud_classification_device` | `str`                       | `"cpu"`                                      | On which device to run cloud classification. Either `"cpu"` or `"cuda"`.                                                                                                                                                                                                                                                                                  |
| `S2_return_cloud_probabilities`  | `bool`                      | `False`                                      | Whether to return raw cloud probabilities which were used to determine the cloud classes.                                                                                                                                                                                                                                                                 |
| `S2_nbar`                        | `bool`                      | `False`                                      | Whether to apply Nadir BRDF (Bidirectional Reflectance Distribution Function) correction to Sentinel-2 surface reflectance using the [sen2nbar](https://github.com/cosminpopescu/sen2nbar) package. This correction harmonizes reflectance values as if observed from nadir, reducing angular effects and improving consistency for time series analysis. |
| `num_workers`                    | `int`                       | `1`                                          | Number of cores to scale computation across. Plan 2GiB of RAM per worker. -1 uses all available cores.                                                                                                                                                                                                                                                    |
| `time_composite_freq`            | `str`                       | `None`                                       | Rounding interval across which data is aggregated.                                                                                                                                                                                                                                                                                                        |
| `time_composite_method`          | `str`                       | `"mean"`                                     | How to aggregate the acquisitions within each `time_composite_freq` window: `"mean"` (default), `"median"`, `"min"` or `"max"`. Applies to both Sentinel-2 and Sentinel-1. NoData/masked pixels are ignored per band and pixel. Only used when `time_composite_freq` is set.                                                                                |
| `S2_apply_snow_mask`             | `bool`                      | `False`                                      | Whether to replace snow with NaN.                                                                                                                                                                                                                                                                                                                         |
| `S2_apply_cloud_mask`            | `bool`                      | `False`                                      | Whether to replace anything that is not clear sky with NaN.                                                                                                                                                                                                                                                                                               |
| `overwrite`                      | `bool`                      | `False`                                      | Whether to overwrite existing zarr storage.                                                                                                                                                                                                                                                                                                               |
| `zarr_store_chunk_size`          | `dict`                      | `{"time": 10, "x": 250, "y": 250}`           | Chunk sizes for zarr storage. Must contain the keys 'time', 'y', and 'x'. Controls the size of data chunks for efficient storage and retrieval.                                                                                                                                                                                                           |
| `resampling_method`              | `rasterio.enums.Resampling` | `Resampling.nearest`                         | Specifies the resampling method that is used to reproject the raw data into the target CRS. It is recommended to use nearest neighbor to prevent potential issues near cloud edges and dynamic range changes.                                                                                                                                             |
| `save_as_uint16` | `bool` | `False` | When `True` and `S1_assets` is `None`, store Sentinel-2 bands as unsigned 16-bit integers with zeros for nodata. NaNs are rounded and clipped into `[0, 65535]` before saving. |


#### Notes

- If `S2_apply_snow_mask` is set to `True`, `S2_mask_snow` must also be `True`.
- If `S2_apply_cloud_mask` is set to `True`, `S2_cloud_classification` must also be `True`.
- If `time_composite_freq` is set and neither `S2_apply_snow_mask` nor `S2_apply_cloud_mask` is set, a warning will be issued as temporal aggregation may yield useless results for Sentinel-2 data.
- When `S1_assets` is supplied as an empty list, it will be converted to `None`, meaning no Sentinel-1 data will be downloaded.
- Passing `S2_bands=None` or `S2_bands=[]` disables Sentinel-2 entirely (Sentinel-1 only) and requires `S1_assets` to be set — this mirrors how `S1_assets=None`/`[]` disables Sentinel-1. Passing a subset list restricts which Sentinel-2 bands are downloaded.
- The `zarr_store_chunk_size` dictionary must contain the keys 'time', 'y', and 'x'.
- When using cloud or snow masking with temporal composites, the masks will be applied before aggregation.
- To download from CDSE (`provider="cdse"`), first create a free [Copernicus Data Space](https://dataspace.copernicus.eu/) account and generate S3 credentials, then make them available to GDAL/boto3 (e.g. add a `[cdse]` profile to `~/.aws/credentials` and run with `AWS_PROFILE=cdse`). CDSE is Sentinel-2 only.
- GDAL's raster block cache is tunable via the standard `GDAL_CACHEMAX` environment variable (sentle honours it and does not override it), e.g. `GDAL_CACHEMAX=512` (MB) — useful on memory-constrained hosts or when running many `num_workers` (the cache is per-process). In practice it has little effect on CDSE JP2 reads (their decoded tiles are cached inside the open dataset, not GDAL's block cache) but does govern the Planetary Computer COG path; the default (≈5% of RAM) is fine for most machines.

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

## Note

This package is meant for large-scale processing and any area that is smaller than 8km in width and height will not run faster because of the underlying processing scheme.
