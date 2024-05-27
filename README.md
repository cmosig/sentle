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
    <em>Download Sentinel-1 & Sentinel-2 data cubes of any scale (larger-than-memory) on any machine with integrated cloud
detection, snow masking, harmonization, merging, and temporal composites.</em>
</p>

---
 
## Important Note

1) The model for cloud detection will be made available within the next couple of weeks.
2) This package is in early alpha stage. If you encounter any error, warning, memory issue, etc. please open a GitHub issue with the code to reproduce.

## Prerequisites

If you download larger areas or longer timeseries you'll need to obtain a
subscription key from [Planetary Computer](https://planetarycomputer.microsoft.com/account/request). 
Configure it in your shell with: `export PC_SDK_SUBSCRIPTION_KEY=xxxxyourkeyxxxx`

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

## Quick Tour

**(1) Initiate the `Sentle` class.** This initiates a [dask](https://www.dask.org/) cluster (don't be scared of the word cluster, this can also mean 1 CPU core) in the background. Each worker needs in practice about 3GB RAM in default settings.

```
from sentle.sentle import Sentle
from rasterio.crs import CRS

sen = Sentle(num_workers=3)
```

**(2) Specify which area you want to download, and in which CRS.** 

The below code sets up the dask task graph and saves a lazy dask array internally (`sen.da`). 

The resulting dask array has the shape `(#timesteps, #bands, #pixelsy, #pixelsx)`. There is one timestep for each timestamp where there is data available anywhere within the specified bounding box. This will result in spatially very sparse timesteps and handled internally.

CRS: For local studies, I recommend the local [UTM zone](https://www.dmap.co.uk/utmworld.htm) and pick the [EPSG code](https://docs.up42.com/data/reference/utm). For continental-scale studies, you may want to use EPSG:8857 or EPSG:3857. 

```
sen.process(
    target_crs=CRS.from_string("EPSG:8857"),
    bound_left=931070,
    bound_bottom=6111250,
    bound_right=957630,
    bound_top=6134550,
    target_resolution=10,
    datetime="2023-06-01/2023-06-14",
    S2_mask_snow=True,
    S2_cloud_classification=True,
    S2_cloud_classification_device="cuda",
    S1_assets=["vv", "vh"])
```
<p align="center">
<img src="https://github.com/cmosig/sentle/assets/32590522/1da22165-9fef-480f-8643-88ba58c18574" width="600">
</p>

**(3) (Optional) Mask out clouds and snow (extends task graph).** 

This removes clouds/snow based on the generated masks, i.e., setting the respective pixels to `nan`.
```
sen.mask_array()
```

**(4) (Optional) Create a time composite in specified time intervals (extends task graph).** 

Creates a (nan)median across each time interval for each S1&S2 band. This will also drop the cloud mask and snow mask if available as these cannot be merged.
If temporal accuracy down to a single day is not relevant to your project, this step is highly recommended. The `freq` argument is passed to [pandas.Timestamp.round](https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.round.html).
```
sen.create_time_composite(freq="14d")
```

Note: this step is not stable at the moment, likely because it is working across chunks. It works fine for <1GiB cubes, but may strain memory with larger cubes.  

**(5) Save to zarr.**
This executes the built-up dask graph and saves the data to an optimized Zarr format.  
```
sen.save_as_zarr("my_cube.zarr")
```
Alternatively, you can call `sen.da.compute()` and use the generated cube directly, without saving it to your drive.

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
