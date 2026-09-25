"""Tests for the per-job cloud-response-queue registry (issue #87).

Each Sentinel-2 ptile gets its own manager ``Queue`` so the cloud-prediction
service can answer it; ``GLOBAL_QUEUES`` holds the parent-side reference that
keeps the manager referent alive while the proxy is in flight to the worker.
The ``cleanupqueue`` joblib backend is supposed to drop that reference once the
ptile is done -- but it only overrode ``apply_async``, which joblib >= 1.5 no
longer dispatches through (``PoolManagerMixin.submit`` shadows the deprecation
shim in the MRO), so the hook silently stopped running and the queues piled up
for the whole run.

The hook runs inside ``multiprocessing.pool.ApplyResult._set`` before
``_event.set()``, and ``Pool._handle_results`` swallows ``KeyError`` there, so
anything it raises fails silently -- hence the spy fixture.

Offline: plain sentinel objects, no manager, no network, no model.
"""

import pytest
from joblib import Parallel, delayed, parallel_backend

import sentle.sentle as sentle_mod
import sentle.utils as sentle_utils


# module level so the fork pool can pickle them
def _echo_job_id(job_id):
    return job_id


def _boom(_):
    raise ValueError("boom")


@pytest.fixture
def registry():
    sentle_utils.GLOBAL_QUEUES.clear()
    yield sentle_utils.GLOBAL_QUEUES
    sentle_utils.GLOBAL_QUEUES.clear()


@pytest.fixture
def cleanup_errors(monkeypatch):
    seen = []
    original = sentle_utils.ImmediateResultBackend.callback

    def spy(self, result):
        try:
            original(self, result)
        except BaseException as exc:  # pragma: no cover -- guarded in callback
            seen.append(exc)
            raise

    monkeypatch.setattr(sentle_utils.ImmediateResultBackend, "callback", spy)
    return seen


def test_backend_drains_the_registry(registry, cleanup_errors):
    for job_id in range(8):
        registry[job_id] = object()

    with parallel_backend("cleanupqueue"):
        out = Parallel(n_jobs=2,
                       batch_size=1)(delayed(_echo_job_id)(i)
                                     for i in range(8))

    assert out == list(range(8))
    assert cleanup_errors == []
    assert registry == {}


def test_backend_tolerates_none_job_ids(registry, cleanup_errors):
    # job_id is None whenever S2_cloud_classification is off
    registry["untouched"] = object()

    with parallel_backend("cleanupqueue"):
        out = Parallel(n_jobs=2,
                       batch_size=1)(delayed(_echo_job_id)(None)
                                     for _ in range(4))

    assert out == [None] * 4
    assert cleanup_errors == []
    assert set(registry) == {"untouched"}


def test_backend_surfaces_the_task_error_not_a_cleanup_error(
        registry, cleanup_errors):
    # on failure the callback receives the rebuilt exception, not a list
    registry[0] = object()

    with pytest.raises(ValueError, match="boom"):
        with parallel_backend("cleanupqueue"):
            Parallel(n_jobs=2, batch_size=1)(delayed(_boom)(i)
                                             for i in range(2))

    assert cleanup_errors == []


def test_registry_stays_shared_with_sentle_utils():
    # process() must clear the registry in place: rebinding the name would only
    # rebind it in sentle.sentle and permanently detach it from the dict the
    # backend hook pops from
    assert sentle_mod.GLOBAL_QUEUES is sentle_utils.GLOBAL_QUEUES
    assert "GLOBAL_QUEUES" not in sentle_mod.process.__code__.co_varnames
