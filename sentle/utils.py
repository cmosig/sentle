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


class ImmediateResultBackend(MultiprocessingBackend):

    def callback(self, result):
        global GLOBAL_QUEUES
        GLOBAL_QUEUES.pop(result[0])

    def apply_async(self, func, callback=None):
        cbs = MultiCallback(callback, self.callback)
        return super().apply_async(func, cbs)


register_parallel_backend('cleanupqueue', ImmediateResultBackend)
