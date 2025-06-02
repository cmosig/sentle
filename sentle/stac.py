import planetary_computer
import pystac_client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

from .const import STAC_ENDPOINT


def get_stac_api_io():
    """
    Returns a StacApiIO object with a retry policy that retries on 502, 503, 504
    with exponential backoff to handle server overload
    """
    retry = Retry(total=15,
                  backoff_factor=1.0,
                  backoff_jitter=0.2,
                  backoff_max=120,
                  status_forcelist=[502, 503, 504],
                  allowed_methods=None)
    return StacApiIO(max_retries=retry)


def open_catalog():
    return pystac_client.Client.open(STAC_ENDPOINT,
                                     modifier=planetary_computer.sign_inplace,
                                     stac_io=get_stac_api_io())
