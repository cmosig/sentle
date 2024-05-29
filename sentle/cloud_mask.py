import os

import numpy as np
import pkg_resources
import torch
import xarray as xr


def load_cloudsen_model(device: str = "cpu"):
    pkg_path = os.path.dirname(
        pkg_resources.resource_filename("sentle", "sentle.py"))
    model_path = os.path.join(pkg_path, "data", "cloudmodel.pt")
    cloudsen_model = torch.jit.load(model_path)
    cloudsen_model.eval()
    cloudsen_model.to(device)

    return cloudsen_model


S2_cloud_mask_band = "S2_cloud_classification"
S2_cloud_prob_bands = [
    "S2_clear_sky_probability", "S2_thick_cloud_probability",
    "S2_thin_cloud_probability", "S2_shadow_probability"
]


def compute_cloud_mask(array: np.array, model: torch.jit.ScriptModule,
                       S2_cloud_classification_device: str):

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
    tensor = tensor.to(S2_cloud_classification_device)

    # Compute the cloud mask
    with torch.no_grad():
        cloud_probabilities = model(tensor.type(torch.float32)).cpu().numpy()

    # remove padding again
    cloud_probabilities = cloud_probabilities[0, :, 2:-2, 2:-2]

    return cloud_probabilities
