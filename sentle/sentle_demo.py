from sentle import process

if __name__ == "__main__":
    print("------------------------------------")
    print("------------------------------------")
    print("------------------------------------")

    # x = process(
    #     target_crs=CRS.from_string("EPSG:8857"),
    #     bound_left=767300,
    #     bound_bottom=7290000,
    #     bound_right=776000,
    #     bound_top=7315000,
    #     datetime="2023-11-16",
    #     # datetime="2023-11-11/2023-12-01",
    #     # datetime="2023-11",
    #     # datetime="2020/2023",
    #     processing_tile_size=4000,
    #     target_resolution=10,
    #     zarr_path="bigout_parallel_test_5.zarr",
    #     num_workers=1,
    #     threads_per_worker=1,
    #     # less then 3GB per worker will likely not work
    #     memory_limit_per_worker="8GB",
    #     mask_snow=True,
    #     return_cloud_probabilities=False,
    #     cloud_classification=False,
    #     compute_nbar=False,
    #     mask_clouds_device="cuda")

    x = process(
        target_crs=CRS.from_string("EPSG:8857"),
        bound_left=931070,
        bound_bottom=6111250,
        bound_right=957630,
        bound_top=6134550,
        datetime="2023-06-10",
        # datetime="2023-06-01/2023-12-01",
        # datetime="2023-11",
        # datetime="2020/2023",
        processing_tile_size=4000,
        target_resolution=10,
        zarr_path="halle_leipzig_6.zarr",
        num_workers=1,
        threads_per_worker=1,
        # less then 3GB per worker will likely not work
        memory_limit_per_worker="8GB",
        mask_snow=True,
        return_cloud_probabilities=False,
        cloud_classification=True,
        compute_nbar=False,
        mask_clouds_device="cuda")

    # x = process(
    #     target_crs=CRS.from_string("EPSG:8857"),
    #     bound_left=921070,
    #     bound_bottom=6101250,
    #     bound_right=977630,
    #     bound_top=6144550,
    #     datetime="2023-06-10",
    #     # datetime="2023-06-01/2023-12-01",
    #     # datetime="2023-11",
    #     # datetime="2020/2023",
    #     processing_tile_size=4000,
    #     target_resolution=10,
    #     zarr_path="/net/scratch/cmosig/halle_leipzig_5.zarr",
    #     num_workers=50,
    #     threads_per_worker=1,
    #     # less then 3GB per worker will likely not work
    #     memory_limit_per_worker="8GB",
    #     mask_snow=False,
    #     return_cloud_probabilities=False,
    #     cloud_classification=False,
    #     compute_nbar=False,
    #     mask_clouds_device="cuda")

    # x = process(
    #     target_crs=CRS.from_string("EPSG:8857"),
    #     bound_left=564670,
    #     bound_bottom=5718050,
    #     bound_right=1084500,
    #     bound_top=6409170,
    #     # datetime="2023-11-11/2023-12-01",
    #     datetime="2023",
    #     processing_tile_size=4000,
    #     target_resolution=10,
    #     zarr_path="bigout_oneday.zarr")
