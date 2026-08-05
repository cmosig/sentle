import contextlib

import joblib
from joblib import register_parallel_backend
from joblib._parallel_backends import MultiprocessingBackend

GLOBAL_QUEUE_MANAGER = None
GLOBAL_QUEUES = dict()


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    # credits:
    # https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class MultiCallback:

    def __init__(self, *callbacks):
        self.callbacks = [cb for cb in callbacks if cb]

    def __call__(self, out):
        for cb in self.callbacks:
            cb(out)


def release_job_queues(result):
    """Release the per-job cloud response queues owned by a finished batch.

    ``result`` is the raw payload a ``multiprocessing.Pool`` callback receives:
    the list of the batch's return values on success, or the (rebuilt)
    exception instance on failure -- joblib's ``_TracebackCapturingWrapper``
    *returns* the exception instead of raising it, so the success callback is
    the one that fires. ``process_ptile`` returns its ``job_id``, which is
    ``None`` when cloud classification is disabled.

    Dropping the entry here releases the manager queue as soon as its ptile is
    done. The registry itself is load-bearing and must not be removed: it holds
    the only strong reference that keeps the manager referent alive while the
    proxy is in flight to the worker.
    """
    if not isinstance(result, (list, tuple)):
        return
    for job_id in result:
        if job_id is not None:
            GLOBAL_QUEUES.pop(job_id, None)


class ImmediateResultBackend(MultiprocessingBackend):

    def callback(self, result):
        # Runs in the pool's result-handler thread, inside
        # ``multiprocessing.pool.ApplyResult._set`` and *before*
        # ``self._event.set()``. An escaping exception strands the job --
        # ``Pool._handle_results`` even swallows ``KeyError`` there -- so never
        # let one out.
        try:
            release_job_queues(result)
        except Exception:
            pass

    def submit(self, func, callback=None):
        # joblib >= 1.5 dispatches through ``submit``. ``PoolManagerMixin``
        # provides it and shadows ``ParallelBackendBase``'s deprecation shim in
        # the MRO, so overriding only ``apply_async`` (as this class used to)
        # silently stopped running the cleanup.
        return super().submit(func, MultiCallback(callback, self.callback))

    def apply_async(self, func, callback=None):
        # joblib <= 1.4 dispatches through ``apply_async`` instead. joblib's own
        # callback must stay first in both entry points: running the cleanup
        # before ``BatchCompletionCallBack`` deadlocks the run.
        return super().apply_async(func, MultiCallback(callback,
                                                       self.callback))


register_parallel_backend('cleanupqueue', ImmediateResultBackend)
