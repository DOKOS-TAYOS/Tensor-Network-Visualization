import matplotlib

matplotlib.use("Agg")

import pytest

from tensor_network_viz.tenpy import plot_tenpy_network_2d, plot_tenpy_network_3d

tenpy = pytest.importorskip("tenpy")

pytestmark = pytest.mark.filterwarnings("ignore:unit_cell_width.*:UserWarning")


def test_real_tenpy_mps_renders() -> None:
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(3)]
    network = MPS.from_product_state(sites, ["up"] * 3, bc="finite")

    fig2d, ax2d = plot_tenpy_network_2d(network)
    fig3d, ax3d = plot_tenpy_network_3d(network)

    assert fig2d is ax2d.figure
    assert fig3d is ax3d.figure
    assert ax3d.name == "3d"
