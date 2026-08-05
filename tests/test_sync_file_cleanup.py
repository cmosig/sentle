"""Tests for removing the zarr write-lock sync file (issue #87).

``setup_zarr_storage`` mints ``/tmp/sentle_<time>.lock`` whenever the zarr chunk
size does not match the processing chunk size, and workers serialize their
writes on it. The cleanup at the end of ``process()`` called
``shutil.rmtree`` on it -- but it is a plain file, so that always raised
``NotADirectoryError`` into a bare ``except`` and every run leaked one lock
file (thousands accumulate in /tmp over weeks).

Offline: ``tmp_path`` only, no network.
"""

import shutil
from pathlib import Path

import pytest
from filelock import FileLock

from sentle import sentle as sentle_mod
from sentle.sentle import cleanup_sync_file


def test_rmtree_cannot_delete_a_lock_file(tmp_path):
    # pins the exact bug this replaced. The file is created directly rather than
    # through FileLock because whether the lock file survives release is
    # version dependent (filelock >= 3.0.6 deliberately keeps it, see
    # tox-dev/filelock#31) -- what is not version dependent is that the path is
    # a file, and shutil.rmtree can never remove one
    lock_path = tmp_path / "sentle_1.lock"
    lock_path.touch()

    assert lock_path.is_file() and not lock_path.is_dir()
    with pytest.raises(NotADirectoryError):
        shutil.rmtree(str(lock_path))
    assert lock_path.exists()

    cleanup_sync_file(str(lock_path))
    assert not lock_path.exists()


def test_cleanup_removes_an_existing_lock_file(tmp_path):
    lock_path = tmp_path / "sentle_2.lock"
    with FileLock(str(lock_path)):
        pass

    cleanup_sync_file(str(lock_path))
    assert not lock_path.exists()


def test_cleanup_tolerates_a_never_created_lock_file(tmp_path):
    # no ptile had data -> no worker ever acquired the lock -> no file
    lock_path = tmp_path / "never_created.lock"
    cleanup_sync_file(str(lock_path))
    assert not lock_path.exists()


def test_cleanup_tolerates_none():
    # setup_zarr_storage returns None when the chunk sizes already match
    cleanup_sync_file(None)


def test_cleanup_swallows_unlink_errors(tmp_path, monkeypatch):
    # a read-only or foreign tmpdir must not fail an otherwise finished run
    lock_path = tmp_path / "sentle_3.lock"
    lock_path.touch()

    def boom(_path):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(sentle_mod, "remove", boom)
    cleanup_sync_file(str(lock_path))


def test_sync_file_path_is_a_plain_file_path(tmp_path):
    from rasterio.crs import CRS

    import pandas as pd

    sync_file_path = sentle_mod.setup_zarr_storage(
        zarr_store=str(tmp_path / "cube.zarr"),
        timestamp_list=[{
            "collection": "sentinel-2-l2a",
            "ts": pd.Timestamp("2023-06-01T10:00:00Z")
        }],
        height=10,
        width=10,
        bound_left=600000,
        bound_right=600100,
        bound_top=5100000,
        bound_bottom=5099900,
        target_resolution=10,
        processing_spatial_chunk_size=4000,
        zarr_store_chunk_size={
            "time": 10,
            "y": 250,
            "x": 250
        },
        S2_bands_to_save=["B02"],
        total_bands_to_save=["B02"],
        target_crs=CRS.from_epsg(32632),
        consolidate_metadata=False,
    )

    assert sync_file_path is not None and sync_file_path.endswith(".lock")
    assert not Path(sync_file_path).exists()  # not created until acquired
    with FileLock(sync_file_path):
        assert Path(sync_file_path).is_file()

    cleanup_sync_file(sync_file_path)
    assert not Path(sync_file_path).exists()
