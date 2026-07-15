"""Guards sentle's ``pkg_resources`` -> ``importlib.resources`` migration.

setuptools >= 81 no longer ships ``pkg_resources`` (it was deprecated and then
removed), so ``import pkg_resources`` raises ``ModuleNotFoundError`` on a fresh
install. sentle must locate its bundled data -- the Sentinel-2 grid and the
cloud-detection model -- via the standard library instead.
"""

import pathlib
from importlib.resources import files

import pytest

import sentle


@pytest.mark.parametrize("relpath", [
    "data/sentinel2_grid_stripped_with_epsg.gpkg",
    "data/cloudmodel.pt",
])
def test_bundled_data_resolves(relpath):
    # the files sentle loads at runtime must be discoverable via importlib
    target = files("sentle").joinpath(relpath)
    assert target.is_file(), f"missing bundled resource: {relpath}"


def test_source_does_not_import_pkg_resources():
    # a source-level guard: pkg_resources is gone on setuptools >=81, so no
    # sentle module may reference it (a transitive dependency still might,
    # which is why we check the source rather than sys.modules).
    src_dir = pathlib.Path(sentle.__file__).parent
    offenders = [
        str(py.relative_to(src_dir)) for py in src_dir.rglob("*.py")
        if "pkg_resources" in py.read_text()
    ]
    assert not offenders, f"pkg_resources still referenced in: {offenders}"
