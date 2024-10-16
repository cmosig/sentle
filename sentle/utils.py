from rasterio import transform
import contextlib
import joblib

def bounds_from_transform_height_width_res(tf, height, width, resolution):
    # minx, miny, maxx, maxy
    return (tf.c, tf.f - (height * resolution), tf.c + (width * resolution),
            tf.f)


def transform_height_width_from_bounds_res(left, bottom, right, top, res):
    width, rem = divmod(right - left, res)
    assert rem == 0
    width = int(width)
    height, rem = divmod(top - bottom, res)
    assert rem == 0
    height = int(height)
    tf = transform.from_bounds(west=left,
                               south=bottom,
                               east=right,
                               north=top,
                               width=width,
                               height=height)

    return tf, height, width

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

