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

---

# SENTLE

Download Sentinel-2 data cubes of any scale (larger-than-memory) on any machine with integrated cloud
detection, snow masking, harmonization, merging, and temporal composites. 

### Prerequisites

If you download larger areas or longer timeseries you'll need to obtain a
subscription key from [Planetary Computer](https://planetarycomputer.microsoft.com/account/request). 
Configure it in your shell with: `export PC_SDK_SUBSCRIPTION_KEY=xxxxyourkeyxxxx`

### Installing

```
pip install sentle
```
or 
```
git clone git@github.com:cmosig/sentle.git
cd sentle
pip install -e .
```

### Quick Tour

Using the package involves a 3-5 step process. 

**(1) Initiate the `Sentle` class.** This initiates a [dask](https://www.dask.org/) cluster (don't be scared of the word cluster, this can also mean 1 CPU core) in the background. Each worker needs at most 8GB RAM in default settings. 

```
sen = Sentle(num_workers=3)
```

**(2) Specify which area you want to download, and in which CRS.** 

For local studies, I recommend the local [UTM zone](https://www.dmap.co.uk/utmworld.htm). For continental-scale studies, you may want to use EPSG:8857 or EPSG:3857. The below code sets up the dask task graph and saves a lazy dask array internally (`sen.da`). 

The resulting dask array has the shape `(#timesteps, #bands, #pixelsy, #pixelsx)`. There is one timestep for each timestamp where there is data available anywhere within the specified bounding box. This will result in spatially very sparse timesteps.
```
sen.process(
    target_crs=CRS.from_string("EPSG:8857"),
    bound_left=931070,
    bound_bottom=6111250,
    bound_right=957630,
    bound_top=6134550,
    datetime="2023-07-01/2023-07-14",
    mask_snow=True,
    cloud_classification=True,
    cloud_classification_device="cuda")
```
**(3) (Optional) Mask out clouds and snow (extends task graph).** 

This removes clouds/snow based on the generated masks, i.e., setting the respective pixels to `nan`.
```
sen.mask_array()
```

**(4) (Optional) Create a time composite in specified time intervals (extends task graph).** 

Creates a (nan)mean across each time interval for each band. 
If temporal accuracy down to a single day is not relevant to your project, this step is highly recommended.
```
sen.create_time_composite(ndays=7)
```

**(5) Save to zarr.**
This executes the built-up dask graph and saves the data to an optimized Zarr format.  
```
sen.save_as_zarr("my_cube.zarr")
```
Alternatively, you can call `sen.da.compute()` and use the generated cube directly, without saving it to your drive.

## Contributing

Please submit issues or pull requests if you feel like something is missing or
needs to be fixed. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Thank you to [David Montero](https://github.com/davemlz) for all the
discussions and his awesome packages which inspired this.
