from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from matplotlib import image as mpimg

_README_GALLERY_RAW_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/DOKOS-TAYOS/Tensor-Network-Visualization/main/images/tensor_network_visualization_logo.png",
    "https://raw.githubusercontent.com/DOKOS-TAYOS/Tensor-Network-Visualization/main/images/gallery/cubic_peps_3d.png",
    "https://raw.githubusercontent.com/DOKOS-TAYOS/Tensor-Network-Visualization/main/images/gallery/mera_2d.png",
    "https://raw.githubusercontent.com/DOKOS-TAYOS/Tensor-Network-Visualization/main/images/gallery/tubular_grid_3d.png",
    "https://raw.githubusercontent.com/DOKOS-TAYOS/Tensor-Network-Visualization/main/images/gallery/network_controls.png",
    "https://raw.githubusercontent.com/DOKOS-TAYOS/Tensor-Network-Visualization/main/images/gallery/tensor_elements_phase.png",
    "https://raw.githubusercontent.com/DOKOS-TAYOS/Tensor-Network-Visualization/main/images/gallery/tensor_elements_controls.png",
)

_DOC_GALLERY_RELATIVE_PATHS: tuple[str, ...] = (
    "../images/gallery/cubic_peps_3d.png",
    "../images/gallery/tensor_elements_phase.png",
    "../images/gallery/tubular_grid_3d.png",
    "../images/gallery/network_controls.png",
    "../images/gallery/tensor_elements_controls.png",
)

_EXAMPLES_GALLERY_RELATIVE_PATHS: tuple[str, ...] = (
    "../images/gallery/cubic_peps_3d.png",
    "../images/gallery/mera_2d.png",
    "../images/gallery/tubular_grid_3d.png",
    "../images/gallery/network_controls.png",
    "../images/gallery/tensor_elements_phase.png",
    "../images/gallery/tensor_elements_controls.png",
)


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_uses_absolute_github_raw_urls_for_logo_and_gallery_images() -> None:
    readme_text = Path("README.md").read_text(encoding="utf-8")

    for raw_url in _README_GALLERY_RAW_URLS:
        assert raw_url in readme_text


def test_guide_and_examples_reference_gallery_assets() -> None:
    guide_text = Path("docs/guide.md").read_text(encoding="utf-8")
    examples_text = Path("examples/README.md").read_text(encoding="utf-8")

    for relative_path in _DOC_GALLERY_RELATIVE_PATHS:
        assert relative_path in guide_text
    for relative_path in _EXAMPLES_GALLERY_RELATIVE_PATHS:
        assert relative_path in examples_text


def test_generate_doc_images_script_declares_expected_outputs() -> None:
    module = _load_module(Path("scripts/generate_doc_images.py"), "generate_doc_images")

    expected_paths = (
        Path("images/gallery/cubic_peps_3d.png"),
        Path("images/gallery/mera_2d.png"),
        Path("images/gallery/tubular_grid_3d.png"),
        Path("images/gallery/network_controls.png"),
        Path("images/gallery/tensor_elements_phase.png"),
        Path("images/gallery/tensor_elements_controls.png"),
    )

    assert expected_paths == module.DOC_GALLERY_RELATIVE_PATHS


def test_generate_doc_images_writes_gallery_files(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tensorkrowch")
    pytest.importorskip("quimb.tensor")

    module = _load_module(Path("scripts/generate_doc_images.py"), "generate_doc_images_run")

    generated = module.generate_doc_images(output_root=tmp_path)

    assert generated
    assert (
        tuple(path.relative_to(tmp_path) for path in generated) == module.DOC_GALLERY_RELATIVE_PATHS
    )
    assert all(path.is_file() for path in generated)

    transparent_names = {
        "cubic_peps_3d.png",
        "mera_2d.png",
        "tubular_grid_3d.png",
        "tensor_elements_phase.png",
    }
    for path in generated:
        image = mpimg.imread(path)
        if path.name in transparent_names:
            assert image.shape[-1] == 4
            assert (image[..., 3] < 1.0).any()
        else:
            if image.shape[-1] == 4:
                assert (image[..., 3] == 1.0).all()
