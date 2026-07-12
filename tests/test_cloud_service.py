"""Tests for the cloud-prediction service IPC (issue #32).

The service hands tiles to a separate process via ``multiprocessing.shared_memory``
instead of pickling the ~26 MB array through the (manager) queue. These tests
run the real service on CPU (the model ships with the package -- no network) and
check that the shared-memory round-trip returns exactly what a direct
``compute_cloud_mask`` call would, across several sequential requests (so a
leaked/renamed block would surface).
"""

import warnings

import numpy as np
import pytest

from sentle import cloud_mask


@pytest.fixture(scope="module")
def model():
    return cloud_mask.load_cloudsen_model("cpu")


@pytest.fixture
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
