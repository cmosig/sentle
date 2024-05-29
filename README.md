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
</p>
<p align="center">
    <em>Download Sentinel-1 & Sentinel-2 data cubes of huge-scale (larger-than-memory) on any machine with integrated cloud
detection, snow masking, harmonization, merging, and temporal composites.</em>
</p>

---
 
## Important Note

1) The model for cloud detection will be made available within the next couple of weeks.
2) This package is in early alpha stage. If you encounter any error, warning, memory issue, etc. please open a GitHub issue with the code to reproduce.

## Installing

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

**(1) Setup**

There is only one important function: `process`. Here, you specify all parameters and the function returns a lazy [dask](https://www.dask.org/) array with the shape `(#timesteps, #bands, #pixelsy, #pixelsx)`.

```
from sentle import sentle
from rasterio.crs import CRS

da = sentle.process(
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
    num_workers=7,
)
```
This code downloads data for a 40km by 40km area with one year of both Sentinel-1 and Sentinel-2. Clouds and snow are detected and replaced with NaNs. Data is also averaged every 7 days. A lazy dask array is returned:

<p align="center">
<img src="https://github.com/cmosig/sentle/assets/32590522/f487bba1-3c10-42a2-9b10-356ab2b44825" width="600">
</p>

Explanation:
- `target_crs`: Specifies the target CRS that all data will be reprojected to.
- `target_resolution`:  Determines the resolution that all data is reprojected to in the `target_crs`. 
- `bound_*`: Bounds in `target_crs` of the area you want to download. Undefined behavior if difference between opposite bounds is not divisable by `target_resolution`.
- `datetime`: Time range that will be downloaded.
- `S2_mask_snow`: Whether to compute snow mask for Sentinel-2 data.
- `S2_cloud_classification`: Whether to a cloud classification layer for Sentinel-2 data.
- `S2_cloud_classification_device`: Where to run cloud classification. If you have an Nvidia GPU then pass `cuda` otherwise `cpu`(default).
- `S2_apply_*`: Whether to apply the respective mask, i.e., replace values by NaN.
- `S1_assets`: Which Sentinel-1 assets to download. Disable Sentinel-1 by setting this to `None`.
- `time_composite_freq`: Rounding interval across which data is averaged. Uses `pandas.Timestamp.round(time_composite_freq)`. Cloud/snow masks are dropped after masking because they cannot be aggregated.
- `num_workers`: Number of cores to use. Plan about 4 GiB of memory usage per worker.

**(2) Compute**

You either run `.compute()` on the returned dask array or pass the object to
`sentle.save_as_zarr(da, path="..."))` which setups zarr storage and saves each chunk as to disk as
soon as it's ready. The latter enables an area and temporal range to be
computed that is much larger than the RAM on your machine. 

**(3 Visualize)**

Load the data with xarray and visualize using for example the awesome [lexcube](https://github.com/msoechting/lexcube) package. Here, band B02 is visualized from the above example. One is able to spot the cloud gaps and the spotty coverage during winter.

<p align="center">
<img src=https://github.com/cmosig/sentle/assets/32590522/33b7f6a0-532e-453b-80db-748d99e753a2/>
</p>  

## Questions you may have

#### Where can I watch the progress of the download?
Upon class initialization, `sentle` prints a link to a [dask dashboard](https://docs.dask.org/en/latest/dashboard.html). Check the bottom right pane in the Status tab for a progress bar. 
A variety of other stats are also visible there. If you are working on a remote machine you may need to use [port forwarding](https://help.ubuntu.com/community/SSH/OpenSSH/PortForwarding) to access the remote dashboard.
![image](https://github.com/cmosig/sentle/assets/32590522/c20516b5-7a9e-4e99-953a-9c8325edea7b)


#### How do I scale this program?
Increase the number of workers using the `num_workers` parameter when setting up the `Sentle` class. You should give each worker 6GB of memory, even if it only needs 3GB in practise in default settings.

#### My dask graph is too big, what do I do?
Increase the `processing_spatial_chunk_size` from `4000` to something higher in the `process` function. This will increase spatial chunk sizes, but will also increase worker memory requirements. Increase worker memory with `memory_limit_per_worker` when initiating the `Sentle` object.

## Contributing

Please submit issues or pull requests if you feel like something is missing or
needs to be fixed. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

Thank you to [David Montero](https://github.com/davemlz) for all the
discussions and his awesome packages which inspired this.
