import multiprocessing as mp
import os
import queue as queue_module
import traceback
import warnings
from importlib.resources import files
from multiprocessing import resource_tracker, shared_memory

import numpy as np
import torch

# number of cloudsen output classes (clear/thick/thin/shadow probabilities)
_N_CLOUD_CLASSES = 4

# Wire protocol on the (manager) response queue. The service answers every
# request with exactly one message:
#   True                     -> the result was written into the worker's out_shm
#   (_ERROR, "<traceback>")  -> the request failed and nothing was written
# The failure payload is a plain string, never an exception object: it has to
# pickle through the manager queue whatever raised (a payload that fails to
# pickle would raise inside the service's put() and hang the worker all over
# again), and pickling an exception drops its traceback anyway.
_ERROR = "error"

# How long a worker waits for the service before declaring it dead. The service
# handles requests one at a time, so the worst case is roughly num_workers x
# per-tile inference plus one model load; measured CPU inference for one
# (12, 732, 732) tile is ~1.5 s, so even 64 workers behind a service running an
# order of magnitude slower stay far below this. Override with the environment
# variable SENTLE_CLOUD_MASK_TIMEOUT (seconds; <= 0 disables the timeout).
DEFAULT_CLOUD_MASK_TIMEOUT = 1800.0

# How long the service is willing to block handing an answer back to a worker
# whose response queue (maxsize=1) nobody drains anymore.
_SERVICE_PUT_TIMEOUT = 60.0


class CloudMaskServiceError(RuntimeError):
    """The cloud-prediction service failed, died or stopped answering."""


def _cloud_mask_timeout() -> float:
    """Worker-side response timeout, overridable per run via the environment."""
    raw = os.environ.get("SENTLE_CLOUD_MASK_TIMEOUT")
    if raw is None:
        return DEFAULT_CLOUD_MASK_TIMEOUT
    try:
        return float(raw)
    except ValueError:
        warnings.warn(
            f"cloud_mask_timeout_invalid value={raw!r} "
            f"note=falling_back_to_{DEFAULT_CLOUD_MASK_TIMEOUT}")
        return DEFAULT_CLOUD_MASK_TIMEOUT


S2_cloud_mask_band = "S2_cloud_classification"
S2_cloud_prob_bands = [
    "S2_clear_sky_probability", "S2_thick_cloud_probability",
    "S2_thin_cloud_probability", "S2_shadow_probability"
]


def load_cloudsen_model(device: str):
    model_path = str(files("sentle") / "data" / "cloudmodel.pt")
    cloudsen_model = torch.jit.load(model_path)
    cloudsen_model.eval()
    cloudsen_model.to(device)
    return cloudsen_model


def init_cloud_prediction_service(device: str = "cpu"):
    # create request queue that is passed both to workers and the cloud prediction loop
    queue_manager = mp.Manager()
    request_queue = queue_manager.Queue()

    process = mp.Process(target=cloud_prediction_loop,
                         args=(request_queue, device))
    process.start()

    return queue_manager, request_queue


def _attach_shared_memory(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing shared-memory block created by a worker.

    The worker that created the block owns its lifecycle (it unlinks it), so we
    unregister the block from *this* process's resource_tracker to prevent it
    from also trying to unlink it (which would double-unlink and spam
    ``resource_tracker`` warnings on shutdown).
    """
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return shm


def _handle_request(request, model, device):
    """Serve one request; raises on failure, the caller reports it back."""
    # read the input tile straight out of shared memory (no 26 MB pickle
    # through the queue). The worker owns/unlinks the block.
    in_shm = _attach_shared_memory(request["in_name"])
    try:
        array = np.ndarray(request["in_shape"],
                           dtype=request["in_dtype"],
                           buffer=in_shm.buf).copy()
    finally:
        in_shm.close()

    cloud_probabilities = compute_cloud_mask(array, model,
                                             device).astype(np.float32)

    # write the result into the worker's pre-allocated output block and
    # only signal completion over the queue.
    out_shm = _attach_shared_memory(request["out_name"])
    try:
        out = np.ndarray(request["out_shape"], dtype=np.float32,
                         buffer=out_shm.buf)
        out[:] = cloud_probabilities
    finally:
        out_shm.close()


def cloud_prediction_loop(request_queue: mp.Queue, device: str):
    # TODO implement batching

    # load model. A failure here (missing CUDA, driver error, ...) must not kill
    # this process: it serves *every* worker, and they would all block on their
    # response queue until their timeout expires. Report the failure per request
    # instead, so the run fails immediately and with the real traceback.
    model = None
    load_failure = None
    try:
        model = load_cloudsen_model(device)
    except Exception:
        load_failure = traceback.format_exc()

    while True:
        request = request_queue.get()

        # if None is received, break the loop
        if request is None:
            break

        if load_failure is not None:
            response = (_ERROR, load_failure)
        else:
            try:
                _handle_request(request, model, device)
                response = True
            except Exception:
                # never let one bad tile (CUDA OOM, driver error, a vanished
                # shared-memory block, ...) kill this process -- see above
                response = (_ERROR, traceback.format_exc())

        try:
            request["response_queue"].put(response,
                                          timeout=_SERVICE_PUT_TIMEOUT)
        except Exception:
            # the worker is gone or no longer drains its queue; nothing we can
            # do about it, but keep serving everyone else
            pass


def compute_cloud_mask(array: np.ndarray, model: torch.jit.ScriptModule,
                       device: str):

    assert array.shape == (
        12, 732,
        732), "only supporting shape (12, 732, 732) for cloud masking for now"

    # add padding so that shape is divisable by 16 for cloudsen
    array = np.pad(array, [(0, 0), (2, 2), (2, 2)], "edge")

    # expand one dim because it needs it
    array = np.expand_dims(array, axis=0)

    # Convert array to torch tensor, divide by 10000
    # This mantains the array in [0,1]
    tensor = torch.from_numpy(array) / 10000

    # move to device
    tensor = tensor.to(device)

    # Compute the cloud mask
    with torch.no_grad():
        cloud_logits = model(tensor.type(torch.float32))
        cloud_probabilities = torch.softmax(cloud_logits, dim=1).cpu().numpy()

    # remove padding again
    cloud_probabilities = cloud_probabilities[0, :, 2:-2, 2:-2]

    return cloud_probabilities


def worker_get_cloud_mask(array: np.ndarray,
                          request_queue: mp.Queue,
                          response_queue: mp.Queue,
                          timeout: float = None):
    """Send a tile to the cloud-prediction service and get the class
    probabilities back.

    The (large) arrays are handed over via ``multiprocessing.shared_memory``
    rather than pickled through the (manager) queue -- only small metadata
    (block names, shapes, dtype) crosses the queue. This worker creates and
    owns both the input and output blocks so the service only ever attaches to
    them, avoiding cross-process resource-tracker cleanup issues.

    ``timeout`` (seconds, default ``SENTLE_CLOUD_MASK_TIMEOUT`` or
    ``DEFAULT_CLOUD_MASK_TIMEOUT``) bounds the wait: a service that was killed
    -- by an OOM kill, a segfault in torch, ... -- would otherwise leave this
    worker, and therefore the whole run, blocked forever. Raises
    ``CloudMaskServiceError`` when the service reports a failure or goes away.
    """
    array = np.ascontiguousarray(array, dtype=np.float32)
    out_shape = (_N_CLOUD_CLASSES, array.shape[1], array.shape[2])
    out_nbytes = int(np.prod(out_shape)) * np.dtype(np.float32).itemsize
    if timeout is None:
        timeout = _cloud_mask_timeout()

    in_shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    out_shm = shared_memory.SharedMemory(create=True, size=out_nbytes)
    try:
        # drop a late answer to an earlier, timed-out request so it cannot be
        # mistaken for the answer to this one
        while True:
            try:
                response_queue.get_nowait()
            except queue_module.Empty:
                break

        # copy the tile into shared memory
        np.ndarray(array.shape, dtype=array.dtype,
                   buffer=in_shm.buf)[:] = array

        request_queue.put({
            "in_name": in_shm.name,
            "in_shape": array.shape,
            "in_dtype": str(array.dtype),
            "out_name": out_shm.name,
            "out_shape": out_shape,
            "response_queue": response_queue,
        })

        # wait until the service has written the result into out_shm
        try:
            response = response_queue.get(
                timeout=timeout if timeout and timeout > 0 else None)
        except queue_module.Empty:
            raise CloudMaskServiceError(
                f"no answer from the cloud-prediction service within "
                f"{timeout:g}s -- it most likely died (crash, OOM kill) or is "
                f"stuck. Set SENTLE_CLOUD_MASK_TIMEOUT to raise the limit on a "
                f"very slow machine, or to 0 to wait indefinitely.") from None

        if isinstance(response, tuple) and response[:1] == (_ERROR, ):
            raise CloudMaskServiceError(
                "cloud-prediction service failed on this tile:\n" +
                str(response[1]))

        return np.ndarray(out_shape, dtype=np.float32,
                          buffer=out_shm.buf).copy()
    finally:
        in_shm.close()
        in_shm.unlink()
        out_shm.close()
        out_shm.unlink()
