"""Tests for the cloud-prediction service IPC (issue #32).

The service hands tiles to a separate process via ``multiprocessing.shared_memory``
instead of pickling the ~26 MB array through the (manager) queue. These tests
run the real service on CPU (the model ships with the package -- no network) and
check that the shared-memory round-trip returns exactly what a direct
``compute_cloud_mask`` call would, across several sequential requests (so a
leaked/renamed block would surface).

The failure-path tests at the bottom cover issue #87: the service used to have
no exception guard, so a single bad tile killed the one process serving every
worker, and ``response_queue.get()`` had no timeout, so all of them then blocked
forever.
"""

import os
import time
import warnings
from multiprocessing import shared_memory

import numpy as np
import pytest

from sentle import cloud_mask


@pytest.fixture(scope="module")
def model():
    return cloud_mask.load_cloudsen_model("cpu")


# Module-scoped so the worker process is forked exactly once, up front, before
# the ``model`` fixture initialises torch in this (the parent) process. A
# function-scoped fixture re-forks the worker for the second test -- i.e. after
# torch's thread pools are already running here -- and forking a process that
# has touched torch deadlocks the child (an OpenMP/fork-safety issue, not
# CUDA-specific). It only bites on few-core machines like CI runners, where the
# race resolves the wrong way every time. Real ``process()`` runs are unaffected
# because they fork the service before any torch call.
@pytest.fixture(scope="module")
def service():
    mgr, request_queue = cloud_mask.init_cloud_prediction_service(device="cpu")
    try:
        yield mgr, request_queue
    finally:
        # stop the service loop and tear down the manager
        request_queue.put(None)
        mgr.shutdown()


def _tile(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 4000, size=(12, 732, 732)).astype(np.float32)


def test_shared_memory_roundtrip_matches_direct(service, model):
    mgr, request_queue = service
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(3):  # several requests reuse the one service
            arr = _tile(seed)
            reference = cloud_mask.compute_cloud_mask(arr, model, "cpu")

            response_queue = mgr.Queue(maxsize=1)
            out = cloud_mask.worker_get_cloud_mask(arr, request_queue,
                                                   response_queue)

            assert out.shape == (4, 732, 732)
            assert out.dtype == np.float32
            # identical to a direct in-process inference
            assert np.allclose(out, reference, atol=1e-5)
            # softmax probabilities sum to one per pixel
            assert np.allclose(out.sum(axis=0), 1.0, atol=1e-4)


def test_result_is_a_private_copy(service, model):
    # the returned array must not alias freed shared memory
    mgr, request_queue = service
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response_queue = mgr.Queue(maxsize=1)
        out = cloud_mask.worker_get_cloud_mask(_tile(7), request_queue,
                                               response_queue)
    # writable, finite, and stays valid after the shm blocks were unlinked
    out += 1.0
    assert np.isfinite(out).all()


@pytest.fixture
def created_blocks(monkeypatch):
    # /dev/shm is machine-wide, so a before/after diff of the whole directory is
    # unreliable -- record exactly the blocks this test creates instead
    names = []
    real = shared_memory.SharedMemory

    class Recording(real):

        def __init__(self, name=None, create=False, size=0, **kwargs):
            super().__init__(name=name, create=create, size=size, **kwargs)
            if create:
                names.append(self.name)

    monkeypatch.setattr(cloud_mask.shared_memory, "SharedMemory", Recording)
    return names


def _still_there(names):
    return [name for name in names if os.path.exists("/dev/shm/" + name)]


def test_service_side_failure_raises_and_service_survives(service, model):
    mgr, request_queue = service
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # violates the (12, 732, 732) assert inside compute_cloud_mask, which
        # used to kill the service process outright
        response_queue = mgr.Queue(maxsize=1)
        with pytest.raises(cloud_mask.CloudMaskServiceError) as excinfo:
            cloud_mask.worker_get_cloud_mask(np.zeros((12, 100, 100),
                                                      dtype=np.float32),
                                             request_queue,
                                             response_queue,
                                             timeout=120)
        # the remote traceback comes back with the error
        assert "AssertionError" in str(excinfo.value)

        # ... and the same service still answers the next, valid request
        out = cloud_mask.worker_get_cloud_mask(_tile(11), request_queue,
                                               mgr.Queue(maxsize=1),
                                               timeout=120)
    assert out.shape == (4, 732, 732)
    assert np.allclose(out.sum(axis=0), 1.0, atol=1e-4)


def test_failure_path_does_not_leak_shared_memory(service, created_blocks):
    mgr, request_queue = service
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(cloud_mask.CloudMaskServiceError):
            cloud_mask.worker_get_cloud_mask(np.zeros((12, 64, 64),
                                                      dtype=np.float32),
                                             request_queue,
                                             mgr.Queue(maxsize=1),
                                             timeout=120)
    assert len(created_blocks) == 2
    assert _still_there(created_blocks) == []


def test_timeout_when_nobody_answers(service, created_blocks):
    # models a service that was killed (OOM / segfault): nothing ever reads the
    # request queue, so without a timeout the worker blocks forever
    mgr, _ = service
    dead_request_queue = mgr.Queue()

    started = time.monotonic()
    with pytest.raises(cloud_mask.CloudMaskServiceError, match="within 2s"):
        cloud_mask.worker_get_cloud_mask(np.zeros((12, 8, 8),
                                                  dtype=np.float32),
                                         dead_request_queue,
                                         mgr.Queue(maxsize=1),
                                         timeout=2)
    elapsed = time.monotonic() - started
    assert 1.5 <= elapsed < 30
    assert _still_there(created_blocks) == []


def test_timeout_default_and_environment_override(monkeypatch):
    monkeypatch.delenv("SENTLE_CLOUD_MASK_TIMEOUT", raising=False)
    assert cloud_mask.DEFAULT_CLOUD_MASK_TIMEOUT >= 600
    assert cloud_mask._cloud_mask_timeout(
    ) == cloud_mask.DEFAULT_CLOUD_MASK_TIMEOUT

    monkeypatch.setenv("SENTLE_CLOUD_MASK_TIMEOUT", "12.5")
    assert cloud_mask._cloud_mask_timeout() == 12.5

    # a typo in the env var must not take a multi-hour run down mid-flight
    monkeypatch.setenv("SENTLE_CLOUD_MASK_TIMEOUT", "soon")
    with pytest.warns(UserWarning, match="cloud_mask_timeout_invalid"):
        assert cloud_mask._cloud_mask_timeout(
        ) == cloud_mask.DEFAULT_CLOUD_MASK_TIMEOUT


def test_error_payload_survives_the_manager_queue(service):
    # a payload that fails to pickle would raise inside the service's put() and
    # bring the hang straight back, so the failure message is plain strings
    mgr, _ = service
    queue = mgr.Queue(maxsize=1)
    queue.put((cloud_mask._ERROR, "Traceback (most recent call last): ..."))
    assert queue.get(timeout=5)[0] == cloud_mask._ERROR


def test_stale_response_is_dropped_before_a_new_request(service):
    # a late answer to an earlier, timed-out request must not be mistaken for
    # the answer to this one
    mgr, request_queue = service
    response_queue = mgr.Queue(maxsize=1)
    response_queue.put(True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = cloud_mask.worker_get_cloud_mask(_tile(13),
                                               request_queue,
                                               response_queue,
                                               timeout=120)
    # real inference, not the zeros of an untouched output block
    assert np.allclose(out.sum(axis=0), 1.0, atol=1e-4)
