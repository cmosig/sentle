"""Tests for the per-ptile worker timeout and the teardown path (issue #87).

``process()`` used to call ``Parallel(...)`` without a ``timeout``, so a single
task that never delivered a result -- a worker killed by the OOM killer, or one
blocked forever in libcurl -- left the main thread polling in joblib's
``_retrieve`` indefinitely. ``multiprocessing.Pool`` has no dead-worker
notification, so neither joblib nor sentle could notice.

The teardown after ``Parallel`` was also unprotected, so any failure (including
the new timeout, and Ctrl-C) leaked the cloud-prediction process, the queue
manager and the sync file.

Offline: the joblib tests only sleep, and the ``process()`` tests replace
``retrieve_timestamps`` and ``Parallel`` with stand-ins, so nothing downloads.
"""

import multiprocessing
import time
from pathlib import Path

import pandas as pd
import pytest
from joblib import Parallel, delayed, parallel_backend

import sentle.utils  # noqa: F401 -- registers the "cleanupqueue" backend
from sentle import sentle as sentle_mod


# module level so the fork pool can pickle it
def _sleep(seconds, value=None):
    time.sleep(seconds)
    return value


def test_stalled_worker_raises_instead_of_hanging():
    # note: multiprocessing.TimeoutError is NOT a subclass of the builtin
    # TimeoutError, so `except TimeoutError` would not catch this
    started = time.monotonic()
    with pytest.raises(multiprocessing.TimeoutError):
        with parallel_backend("cleanupqueue"):
            Parallel(n_jobs=2, batch_size=1,
                     timeout=2)(delayed(_sleep)(600 if i == 1 else 0.01)
                                for i in range(4))
    assert time.monotonic() - started < 30


def test_timeout_is_per_ptile_not_wall_clock():
    # the budget is measured from the moment a task reaches the head of the
    # retrieval queue, so a run far longer than the timeout -- and tasks that
    # wait in the pool queue for multiples of it -- must not trip it
    with parallel_backend("cleanupqueue"):
        out = Parallel(n_jobs=2, batch_size=1, timeout=2,
                       pre_dispatch="all")(delayed(_sleep)(1.0, i)
                                           for i in range(10))
    assert out == list(range(10))


class _RecordingParallel:
    """Stands in for joblib's Parallel: records kwargs, runs no worker."""
    last_kwargs = None
    sync_file_path = None

    def __init__(self, **kwargs):
        _RecordingParallel.last_kwargs = kwargs

    def _drain(self, jobs):
        for job in jobs:
            # joblib's delayed() yields (func, args, kwargs)
            sync_file_path = job[2]["sync_file_path"]
            # a worker taking the zarr write lock is what actually creates the
            # sync file, so create it here to exercise the cleanup
            _RecordingParallel.sync_file_path = sync_file_path
            Path(sync_file_path).touch()

    def __call__(self, jobs):
        self._drain(jobs)
        return []


class _TimingOutParallel(_RecordingParallel):

    def __call__(self, jobs):
        self._drain(jobs)
        raise multiprocessing.TimeoutError()


class _FakeQueue:

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _FakeManager:

    def __init__(self):
        self.shutdown_calls = 0

    def Queue(self, maxsize=None):
        return _FakeQueue()

    def shutdown(self):
        self.shutdown_calls += 1


def _offline_process(monkeypatch, tmp_path, parallel_cls, **overrides):
    monkeypatch.setattr(
        sentle_mod, "retrieve_timestamps", lambda **kwargs: [{
            "collection": "sentinel-2-l2a",
            "ts": pd.Timestamp("2023-06-01T10:00:00Z")
        }])
    monkeypatch.setattr(sentle_mod, "Parallel", parallel_cls)
    # the bundled MGRS grid takes ~10s to read and nothing here needs the real
    # subtile geometry -- these tests are about the Parallel wiring and teardown
    monkeypatch.setattr(sentle_mod.gpd, "read_file", lambda *a, **k: None)
    monkeypatch.setattr(sentle_mod, "obtain_subtiles", lambda **kwargs: None)

    kwargs = dict(
        target_crs="EPSG:32632",
        target_resolution=10,
        bound_left=600000,
        bound_bottom=5099900,
        bound_right=600100,
        bound_top=5100000,
        datetime="2023-06-01/2023-06-02",
        zarr_store=str(tmp_path / "cube.zarr"),
        S1_assets=None,
        num_workers=4,
    )
    kwargs.update(overrides)
    return sentle_mod.process(**kwargs)


def test_process_passes_the_timeout_to_parallel(monkeypatch, tmp_path):
    _offline_process(monkeypatch,
                     tmp_path,
                     _RecordingParallel,
                     worker_timeout=123.0)
    assert _RecordingParallel.last_kwargs["timeout"] == 123.0


def test_process_has_a_worker_timeout_by_default(monkeypatch, tmp_path):
    _offline_process(monkeypatch, tmp_path, _RecordingParallel)
    assert _RecordingParallel.last_kwargs["timeout"] is not None


def test_process_disables_the_timeout_for_a_single_worker(
        monkeypatch, tmp_path):
    # joblib falls back to the SequentialBackend at n_jobs == 1, which cannot
    # honour a timeout and warns about it on every run if one is passed
    _offline_process(monkeypatch,
                     tmp_path,
                     _RecordingParallel,
                     num_workers=1,
                     worker_timeout=123.0)
    assert _RecordingParallel.last_kwargs["timeout"] is None


def test_sync_file_is_removed_after_a_successful_run(monkeypatch, tmp_path):
    _offline_process(monkeypatch, tmp_path, _RecordingParallel)

    sync_file_path = _RecordingParallel.sync_file_path
    assert sync_file_path is not None
    assert not Path(sync_file_path).exists()


def test_cleanup_runs_when_a_worker_times_out(monkeypatch, tmp_path):
    manager = _FakeManager()
    request_queue = _FakeQueue()
    monkeypatch.setattr(sentle_mod, "init_cloud_prediction_service",
                        lambda device: (manager, request_queue))

    with pytest.raises(multiprocessing.TimeoutError):
        _offline_process(monkeypatch,
                         tmp_path,
                         _TimingOutParallel,
                         S2_cloud_classification=True)

    # the cloud-prediction service got its shutdown sentinel, the queue manager
    # was shut down and the sync file removed -- all of which used to be skipped
    # whenever anything escaped the Parallel call
    assert request_queue.items == [None]
    assert manager.shutdown_calls == 1
    assert sentle_mod.GLOBAL_QUEUES == {}
    assert not Path(_RecordingParallel.sync_file_path).exists()
