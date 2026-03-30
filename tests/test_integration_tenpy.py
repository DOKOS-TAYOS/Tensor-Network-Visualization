import matplotlib

matplotlib.use("Agg")

import pytest

from tensor_network_viz.tenpy import plot_tenpy_network_2d, plot_tenpy_network_3d

tenpy = pytest.importorskip("tenpy")
from tenpy.tools.misc import BetaWarning  # noqa: E402

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


def test_real_tenpy_infinite_mps_renders() -> None:
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(3)]
    network = MPS.from_product_state(sites, ["up"] * 3, bc="infinite")

    fig2d, ax2d = plot_tenpy_network_2d(network)
    fig3d, ax3d = plot_tenpy_network_3d(network)

    assert fig2d is ax2d.figure
    assert fig3d is ax3d.figure
    assert ax3d.name == "3d"


def test_real_tenpy_infinite_mpo_renders() -> None:
    from tenpy.models.tf_ising import TFIChain

    model = TFIChain({"L": 3, "J": 1.0, "g": 1.0, "bc_MPS": "infinite"})
    network = model.calc_H_MPO()

    fig2d, ax2d = plot_tenpy_network_2d(network)
    fig3d, ax3d = plot_tenpy_network_3d(network)

    assert fig2d is ax2d.figure
    assert fig3d is ax3d.figure
    assert ax3d.name == "3d"


def test_real_tenpy_purification_mps_renders() -> None:
    from tenpy.networks.purification_mps import PurificationMPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(3)]
    network = PurificationMPS.from_infiniteT(sites, bc="finite")

    fig2d, ax2d = plot_tenpy_network_2d(network)
    fig3d, ax3d = plot_tenpy_network_3d(network)

    assert fig2d is ax2d.figure
    assert fig3d is ax3d.figure
    assert ax3d.name == "3d"


@pytest.mark.filterwarnings("ignore", category=BetaWarning)
def test_real_tenpy_uniform_mps_renders() -> None:
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    from tenpy.networks.uniform_mps import UniformMPS

    sites = [SpinHalfSite() for _ in range(3)]
    imps = MPS.from_product_state(sites, ["up"] * 3, bc="infinite")
    network = UniformMPS.from_MPS(imps)

    fig2d, ax2d = plot_tenpy_network_2d(network)
    fig3d, ax3d = plot_tenpy_network_3d(network)

    assert fig2d is ax2d.figure
    assert fig3d is ax3d.figure
    assert ax3d.name == "3d"
