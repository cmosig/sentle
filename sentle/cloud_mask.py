import multiprocessing as mp
import os

import numpy as np
import pkg_resources
import torch

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


def cloud_prediction_loop(request_queue: mp.Queue, device: str):
    # TODO implement batching

    # load model
    model = load_cloudsen_model(device)

    while True:
        request = request_queue.get()

        # if None is received, break the loop
        if request is None:
            break

        cloud_probabilities = compute_cloud_mask(request["array"], model,
                                                 device)
        request["response_queue"].put(cloud_probabilities)


def compute_cloud_mask(array: np.ndarray, model: torch.jit.ScriptModule,
                       device: str):

    assert array.ndim == 3 and array.shape[0] == 12, (
        "cloud masking expects a (12, H, W) array")
    assert array.shape[1] == array.shape[2], (
        "cloud masking expects square subtiles")

    # cloudsen's U-Net downsamples by 32, so the spatial dims must be divisible
    # by 32; pad (symmetrically where possible) up to the next multiple of 32
    # and remove the padding afterwards. e.g. 732 -> 736 (pad 2/2),
    # 244 -> 256 (pad 6/6).
    size = array.shape[1]
    pad_total = (-size) % 32
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    array = np.pad(array, [(0, 0), (pad_before, pad_after),
                           (pad_before, pad_after)], "edge")

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

    # remove padding again (crop back to the original subtile extent)
    cloud_probabilities = cloud_probabilities[0, :,
                                              pad_before:pad_before + size,
                                              pad_before:pad_before + size]

    return cloud_probabilities


def worker_get_cloud_mask(array: np.ndarray, request_queue: mp.Queue,
                          response_queue: mp.Queue):
    request_queue.put({"array": array, "response_queue": response_queue})
    return response_queue.get()
