import multiprocessing as mp
import os
from multiprocessing import resource_tracker, shared_memory

import numpy as np
import pkg_resources
import torch

# number of cloudsen output classes (clear/thick/thin/shadow probabilities)
_N_CLOUD_CLASSES = 4

S2_cloud_mask_band = "S2_cloud_classification"
S2_cloud_prob_bands = [
    "S2_clear_sky_probability", "S2_thick_cloud_probability",
    "S2_thin_cloud_probability", "S2_shadow_probability"
]


def load_cloudsen_model(device: str):
    pkg_path = os.path.dirname(
        pkg_resources.resource_filename("sentle", "sentle.py"))
    model_path = os.path.join(pkg_path, "data", "cloudmodel.pt")
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


def cloud_prediction_loop(request_queue: mp.Queue, device: str):
    # TODO implement batching

    # load model
    model = load_cloudsen_model(device)

    while True:
        request = request_queue.get()

        # if None is received, break the loop
        if request is None:
            break

        # read the input tile straight out of shared memory (no 26 MB pickle
        # through the queue). The worker owns/unlinks the block.
        in_shm = _attach_shared_memory(request["in_name"])
        try:
            array = np.ndarray(request["in_shape"],
                               dtype=request["in_dtype"],
                               buffer=in_shm.buf).copy()
        finally:
            in_shm.close()

        cloud_probabilities = compute_cloud_mask(array, model, device).astype(
            np.float32)

        # write the result into the worker's pre-allocated output block and
        # only signal completion over the queue.
        out_shm = _attach_shared_memory(request["out_name"])
        try:
            out = np.ndarray(request["out_shape"], dtype=np.float32,
                             buffer=out_shm.buf)
            out[:] = cloud_probabilities
        finally:
            out_shm.close()

        request["response_queue"].put(True)


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


def worker_get_cloud_mask(array: np.ndarray, request_queue: mp.Queue,
                          response_queue: mp.Queue):
    """Send a tile to the cloud-prediction service and get the class
    probabilities back.

    The (large) arrays are handed over via ``multiprocessing.shared_memory``
    rather than pickled through the (manager) queue -- only small metadata
    (block names, shapes, dtype) crosses the queue. This worker creates and
    owns both the input and output blocks so the service only ever attaches to
    them, avoiding cross-process resource-tracker cleanup issues.
    """
    array = np.ascontiguousarray(array, dtype=np.float32)
    out_shape = (_N_CLOUD_CLASSES, array.shape[1], array.shape[2])
    out_nbytes = int(np.prod(out_shape)) * np.dtype(np.float32).itemsize

    in_shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    out_shm = shared_memory.SharedMemory(create=True, size=out_nbytes)
    try:
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
        response_queue.get()

        return np.ndarray(out_shape, dtype=np.float32,
                          buffer=out_shm.buf).copy()
    finally:
        in_shm.close()
        in_shm.unlink()
        out_shm.close()
        out_shm.unlink()
