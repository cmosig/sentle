import numpy as np
import torch
import xarray as xr
import pkg_resources
import os


def load_cloudsen_model(device: str = "cpu"):
    pkg_path = os.path.dirname(
        pkg_resources.resource_filename("sentle", "sentle.py"))
    model_path = os.path.join(pkg_path, "data", "cloudmodel.pt")
    cloudsen_model = torch.jit.load(model_path)
    cloudsen_model.eval()
    cloudsen_model.to(device)

    return cloudsen_model


def compute_cloud_mask(array: np.array, model: torch.jit.ScriptModule,
                       mask_clouds_device: str):

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
    tensor = tensor.to(mask_clouds_device)

    # Compute the cloud mask
    with torch.no_grad():
        cloud_probabilities = model(tensor.type(torch.float32)).cpu().numpy()

    band_names = [
        "clear_sky_probability",
        "thick_cloud_probability",
        "thin_cloud_probability",
        "shadow_probability",
    ]

    # remove padding again
    cloud_probabilities = cloud_probabilities[0, :, 2:-2, 2:-2]

    return band_names, cloud_probabilities
