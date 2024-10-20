import io
import pickle
from multiprocessing import shared_memory

import geopandas as gpd
import pkg_resources


def load_sentinel_2_grid_into_memory():
    df = gpd.read_file(
        pkg_resources.resource_filename(
            __name__, "data/sentinel2_grid_stripped_with_epsg.gpkg"))

    buffer = io.BytesIO()
    pickle.dump(df, buffer)
    mem = shared_memory.SharedMemory(name=None,
                                     create=True,
                                     size=len(buffer.getvalue()))
    mem.buf[:] = buffer.getvalue()
    return mem.name

    def delete_sentinel_2_grid_from_memory(mem_name):
        shared_memory.SharedMemory(mem_name).close()
        shared_memory.SharedMemory(mem_name).unlink()


def load_sentinel_2_grid_from_memory(mem_name):
    mem = shared_memory.SharedMemory(mem_name, create=False)
    return pickle.load(io.BytesIO(mem.buf))
