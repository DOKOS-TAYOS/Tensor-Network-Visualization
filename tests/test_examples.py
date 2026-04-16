from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from plotting_helpers import assert_readable_image
from tensor_network_viz import EinsumTrace, show_tensor_network
from tensor_network_viz._tensor_elements_data import (
    _extract_einsum_playback_step_records,
    _extract_playback_step_records,
)

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from demo_cli import ExampleCliArgs  # noqa: E402


def _require_quimb() -> None:
    pytest.importorskip("quimb.tensor")


def _require_tenpy() -> None:
    pytest.importorskip("tenpy")


def _require_tensornetwork() -> None:
    pytest.importorskip("tensornetwork")


def _require_torch() -> None:
    pytest.importorskip("torch")


def _require_demo_backend(engine: str) -> None:
    if engine in {"themes", "placements", "geometry", "quimb"}:
        _require_quimb()
    elif engine == "tenpy":
        _require_tenpy()
    elif engine == "tensornetwork":
        _require_tensornetwork()
    elif engine in {"tensorkrowch", "einsum"}:
        _require_torch()
        if engine == "tensorkrowch":
            pytest.importorskip("tensorkrowch")


def _load_example_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_demo_registry_declares_expected_example_sets() -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_registry")

    assert set(module.available_engines()) == {
        "einsum",
        "geometry",
        "placements",
        "quimb",
        "tenpy",
        "tensorkrowch",
        "tensornetwork",
        "themes",
    }
    assert set(module.available_examples("tensorkrowch")) == {
        "cubic_peps",
        "disconnected",
        "ladder",
        "mera",
        "mera_ttn",
        "mps",
        "mpo",
        "peps",
        "weird",
    }
    assert set(module.available_examples("tensornetwork")) == {
        "mps",
        "peps",
        "weird",
    }
    assert set(module.available_examples("quimb")) == {
        "hyper",
        "mps",
        "peps",
    }
    assert set(module.available_examples("tenpy")) == {
        "chain",
        "excitation",
        "hub",
        "hyper",
        "impo",
        "imps",
        "mps",
        "mpo",
        "purification",
        "uniform",
    }
    assert set(module.available_examples("einsum")) == {
        "batch",
        "disconnected",
        "ellipsis",
        "implicit_out",
        "mps",
        "mpo",
        "nway",
        "peps",
        "ternary",
        "trace",
        "unary",
    }
    assert set(module.available_examples("themes")) == {"overview", "tensor_elements"}
    assert set(module.available_examples("placements")) == {
        "grid2d",
        "grid3d",
        "list",
        "manual_positions",
        "manual_scheme",
        "named_indices",
        "object",
    }
    assert set(module.available_examples("geometry")) == {
        "circular_chords",
        "circular_ring",
        "decorated_sparse_grid2d",
        "disconnected_irregular",
        "partial_grid2d",
        "partial_grid3d",
        "random_irregular",
        "tubular_grid",
        "upper_pyramid3d",
        "upper_triangle2d",
    }


def test_run_demo_alias_tt_resolves_to_mps() -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_alias")

    resolved = module.resolve_requested_example(engine="quimb", example="tt")

    assert resolved == "mps"


def test_run_demo_unknown_engine_lists_valid_engines(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_bad_engine")

    with pytest.raises(SystemExit, match="2"):
        module.main(["bad-engine", "mps"])

    captured = capsys.readouterr()
    assert "Unknown engine 'bad-engine'" in captured.err
    assert (
        "einsum, geometry, placements, quimb, tenpy, tensorkrowch, tensornetwork, themes"
        in captured.err
    )


def test_run_demo_unknown_example_lists_valid_examples_for_engine(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_bad_example")

    with pytest.raises(SystemExit, match="2"):
        module.main(["quimb", "bad-example"])

    captured = capsys.readouterr()
    assert "Unknown example 'bad-example' for engine 'quimb'" in captured.err
    assert "hyper" in captured.err
    assert "peps" in captured.err


def test_run_demo_rejects_unsupported_from_list(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_from_list_error")

    with pytest.raises(SystemExit, match="2"):
        module.main(["tenpy", "imps", "--from-list"])

    captured = capsys.readouterr()
    assert "does not support --from-list" in captured.err


def test_run_demo_rejects_unsupported_from_scratch(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_from_scratch_error")

    with pytest.raises(SystemExit, match="2"):
        module.main(["tenpy", "uniform", "--from-scratch"])

    captured = capsys.readouterr()
    assert "does not support --from-scratch" in captured.err


def test_run_demo_auto_save_path_is_used_when_save_has_no_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_auto_save")
    save_paths: list[Path] = []

    def _fake_run_example(args) -> tuple[str, Path]:
        assert args.engine == "einsum"
        assert args.example == "batch"
        assert args.save is not None
        save_paths.append(args.save)
        return ("figure", args.save)

    monkeypatch.setattr(module, "dispatch_run", _fake_run_example)

    exit_code = module.main(["einsum", "batch", "--save", "--no-show"])

    assert exit_code == 0
    assert save_paths == [Path(".tmp") / "examples" / "einsum" / "batch.png"]


def test_quimb_hyper_saves_figure_without_showing() -> None:
    _require_quimb()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_quimb_hyper")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quimb-hyper-demo.png"

    exit_code = module.main(
        ["quimb", "hyper", "--view", "2d", "--save", str(output_path), "--no-show"]
    )

    assert exit_code == 0
    image = assert_readable_image(output_path)
    assert image.shape[0] > 0
    assert image.shape[1] > 0


def test_tenpy_imps_saves_figure_without_showing() -> None:
    _require_tenpy()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_tenpy_imps")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tenpy-imps-demo.png"

    exit_code = module.main(
        ["tenpy", "imps", "--view", "2d", "--save", str(output_path), "--no-show"]
    )

    assert exit_code == 0
    image = assert_readable_image(output_path)
    assert image.shape[0] > 0
    assert image.shape[1] > 0


def test_tenpy_chain_saves_figure_without_showing() -> None:
    _require_tenpy()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_tenpy_chain")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tenpy-chain-demo.png"

    exit_code = module.main(
        ["tenpy", "chain", "--view", "2d", "--save", str(output_path), "--no-show"]
    )

    assert exit_code == 0
    image = assert_readable_image(output_path)
    assert image.shape[0] > 0
    assert image.shape[1] > 0


def test_einsum_ellipsis_saves_figure_without_showing() -> None:
    _require_torch()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_einsum_ellipsis")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "einsum-ellipsis-demo.png"

    exit_code = module.main(
        ["einsum", "ellipsis", "--view", "2d", "--save", str(output_path), "--no-show"]
    )

    assert exit_code == 0
    image = assert_readable_image(output_path)
    assert image.shape[0] > 0
    assert image.shape[1] > 0


@pytest.mark.parametrize(
    ("engine", "example", "view"),
    [
        ("themes", "overview", "2d"),
        ("placements", "manual_scheme", "2d"),
        ("placements", "grid3d", "3d"),
        ("geometry", "decorated_sparse_grid2d", "2d"),
        ("geometry", "upper_triangle2d", "2d"),
        ("geometry", "partial_grid3d", "3d"),
        ("geometry", "disconnected_irregular", "2d"),
        ("tensorkrowch", "weird", "2d"),
        ("tensornetwork", "weird", "2d"),
        ("quimb", "hyper", "2d"),
        ("tenpy", "chain", "2d"),
        ("einsum", "batch", "2d"),
    ],
)
def test_realistic_gallery_demos_save_figures_without_showing(
    engine: str,
    example: str,
    view: str,
) -> None:
    _require_demo_backend(engine)
    module = _load_example_module(
        Path("examples/run_demo.py"),
        f"run_demo_{engine}_{example}_{view}",
    )
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{engine}-{example}-{view}.png"

    exit_code = module.main(
        [engine, example, "--view", view, "--save", str(output_path), "--no-show"]
    )

    assert exit_code == 0
    image = assert_readable_image(output_path)
    assert image.shape[0] > 0
    assert image.shape[1] > 0


@pytest.mark.parametrize("example_name", ["mps", "ellipsis"])
def test_einsum_auto_examples_keep_tensors_alive_for_tensor_inspector(example_name: str) -> None:
    _require_torch()
    run_demo = _load_example_module(Path("examples/run_demo.py"), f"run_demo_{example_name}_args")
    einsum_demo = importlib.import_module("einsum_demo")

    args = run_demo.parse_args(
        ["einsum", example_name, "--view", "2d", "--tensor-inspector", "--no-show"]
    )
    trace = einsum_demo._trace_steps_for(example_name, args)

    assert isinstance(trace, EinsumTrace)
    step_records = _extract_einsum_playback_step_records(trace)
    assert step_records
    assert all(step.record is not None for step in step_records)


@pytest.mark.parametrize("example_name", ["mps", "mpo"])
def test_einsum_auto_examples_keep_inspector_and_costs_aligned_to_real_trace_steps(
    example_name: str,
) -> None:
    _require_torch()
    run_demo = _load_example_module(
        Path("examples/run_demo.py"),
        f"run_demo_{example_name}_inspector_alignment",
    )
    demo_cli = importlib.import_module("demo_cli")
    einsum_demo = importlib.import_module("einsum_demo")

    args = run_demo.parse_args(
        ["einsum", example_name, "--view", "2d", "--tensor-inspector", "--no-show"]
    )
    trace = einsum_demo._trace_steps_for(example_name, args)
    scheme_steps = einsum_demo._scheme_steps(example_name, args)

    assert isinstance(trace, EinsumTrace)
    assert scheme_steps is not None
    assert len(scheme_steps) == len(trace)

    config = demo_cli.finalize_demo_plot_config(
        args,
        engine="einsum",
        scheme_tensor_names=scheme_steps,
    )
    fig, _ax = show_tensor_network(
        trace,
        engine="einsum",
        view=args.view,
        config=config,
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls.current_scene.contraction_controls is not None
    viewer = controls.current_scene.contraction_controls._viewer
    assert viewer is not None
    assert viewer.current_step == len(trace)

    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None
    inspector_controls = getattr(
        inspector._figure,
        "_tensor_network_viz_tensor_elements_controls",
        None,
    )
    assert inspector_controls is not None
    assert f"r{len(trace) - 1}" in inspector_controls._panel.main_ax.get_title()

    controls.cost_hover_on = True
    controls._apply_scene_state(controls.current_scene)
    assert viewer._cost_panel_ax is not None
    assert viewer._cost_panel_ax.get_visible()
    assert viewer._cost_text_artist is not None
    assert "Contraction:" in viewer._cost_text_artist.get_text()


def test_tensornetwork_weird_saves_figure_without_showing() -> None:
    _require_tensornetwork()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_tn_weird")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tensornetwork-weird-demo.png"

    exit_code = module.main(
        [
            "tensornetwork",
            "weird",
            "--view",
            "2d",
            "--save",
            str(output_path),
            "--no-show",
        ]
    )

    assert exit_code == 0
    image = assert_readable_image(output_path)
    assert image.shape[0] > 0
    assert image.shape[1] > 0


def test_commands_doc_lists_copy_paste_demo_commands() -> None:
    text = Path("commands.md").read_text(encoding="utf-8")
    examples_text = Path("examples/README.md").read_text(encoding="utf-8")

    assert "python examples/run_demo.py themes overview" in text
    assert "python examples/run_demo.py themes tensor_elements" in text
    assert "python examples/run_demo.py placements manual_scheme" in text
    assert "python examples/run_demo.py geometry decorated_sparse_grid2d" in text
    assert "python examples/run_demo.py geometry disconnected_irregular" in text
    assert "python examples/run_demo.py quimb hyper" in text
    assert "python examples/run_demo.py geometry partial_grid3d --view 2d" in text
    assert "python examples/run_demo.py geometry upper_pyramid3d --view 2d" in text
    assert "decorated_sparse_grid2d" in examples_text
    assert "partial_grid2d" in examples_text
    assert "partial_grid3d" in examples_text
    assert "listas planas" in examples_text
    assert "layout automatico" in examples_text
    assert "circular_ring" in examples_text
    assert "tubular_grid" in examples_text


def test_placements_manual_positions_3d_saves_without_coordinate_warnings() -> None:
    _require_quimb()
    module = _load_example_module(
        Path("examples/run_demo.py"),
        "run_demo_placements_manual_positions_3d",
    )
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "placements-manual-positions-3d.png"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exit_code = module.main(
            [
                "placements",
                "manual_positions",
                "--view",
                "3d",
                "--save",
                str(output_path),
                "--no-show",
            ]
        )

    assert exit_code == 0
    assert not [
        warning
        for warning in caught
        if "missing coords will be zero-filled" in str(warning.message)
    ]
    image = assert_readable_image(output_path)
    assert image.shape[0] > 0


def test_placements_named_indices_uses_directional_dangling_labels() -> None:
    _require_quimb()
    module = _load_example_module(
        Path("examples/placements_demo.py"),
        "placements_demo_named_indices_labels",
    )

    network = module._named_indices_network()
    index_counts: dict[str, int] = {}
    for tensor in network.tensors:
        for index_name in tensor.inds:
            index_counts[index_name] = index_counts.get(index_name, 0) + 1

    dangling_index_names = {index_name for index_name, count in index_counts.items() if count == 1}
    assert {"left", "up", "front", "down", "right", "out"} <= dangling_index_names


def test_tensorkrowch_run_example_2d_calls_renderer_without_scope_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module(
        Path("examples/tensorkrowch_demo.py"), "tensorkrowch_demo_run_plain"
    )
    from matplotlib.figure import Figure

    def fake_builder(
        args: object,
        definition: object,
    ) -> object:
        del args, definition
        return module.BuiltExample(
            network=object(),
            plot_engine="tensorkrowch",
            title="Title",
        )

    def fake_render_demo_tensor_network(
        network: object,
        *,
        args: object,
        engine: str,
        view: str,
        config: object,
    ) -> tuple[Figure, object]:
        del network, config
        assert args is not None
        assert engine == "tensorkrowch"
        assert view == "2d"
        return Figure(), object()

    args = module.ExampleCliArgs(
        engine="tensorkrowch",
        example="mps",
        view="2d",
        labels_nodes=True,
        labels_edges=False,
        labels=None,
        hover_labels=True,
        scheme=False,
        hover_cost=False,
        tensor_inspector=False,
        contracted=False,
        from_scratch=False,
        from_list=False,
        save=None,
        no_show=True,
        n_sites=2,
        lx=1,
        ly=1,
        lz=1,
        mera_log2=1,
        tree_depth=1,
    )

    fake_definition = module.ExampleDefinition(
        name="mps",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=fake_builder,
        description="fake",
    )

    monkeypatch.setattr(
        module,
        "resolve_example_definition",
        lambda definitions, requested: fake_definition,
    )
    monkeypatch.setattr(module, "render_demo_tensor_network", fake_render_demo_tensor_network)
    monkeypatch.setattr(module, "apply_demo_caption", lambda fig, **kwargs: None)

    module.run_example(args)


def test_tensorkrowch_contracted_demo_uses_native_network_and_auto_scheme() -> None:
    pytest.importorskip("tensorkrowch")
    pytest.importorskip("torch")
    module = _load_example_module(
        Path("examples/tensorkrowch_demo.py"), "tensorkrowch_demo_contracted"
    )
    run_demo = _load_example_module(Path("examples/run_demo.py"), "run_demo_tk_contracted_args")

    args = run_demo.parse_args(
        ["tensorkrowch", "mps", "--contracted", "--scheme", "--n-sites", "6", "--no-show"]
    )
    definition = module.resolve_example_definition(module.EXAMPLES, "mps")
    built = module._build_example(args, definition)

    assert built.plot_engine == "tensorkrowch"
    assert built.scheme_steps_by_name is None
    assert "auto-recovered contraction history" in (built.footer or "")
    assert hasattr(built.network, "resultant_nodes")
    assert len(built.network.resultant_nodes) == 5


def test_tensorkrowch_mps_demo_uses_pairwise_merge_contractions_by_default() -> None:
    pytest.importorskip("tensorkrowch")
    pytest.importorskip("torch")
    module = _load_example_module(
        Path("examples/tensorkrowch_demo.py"),
        "tensorkrowch_demo_pairwise",
    )
    run_demo = _load_example_module(Path("examples/run_demo.py"), "run_demo_tk_pairwise_args")
    from tensor_network_viz.tensorkrowch.graph import _build_graph as _build_tensorkrowch_graph

    args = run_demo.parse_args(["tensorkrowch", "mps", "--scheme", "--n-sites", "6", "--no-show"])
    definition = module.resolve_example_definition(module.EXAMPLES, "mps")
    built = module._build_example(args, definition)

    assert built.scheme_steps_by_name is None
    graph = _build_tensorkrowch_graph(built.network)
    assert graph.contraction_steps is not None
    assert len(graph.contraction_steps) == 5
    assert tuple(len(step) for step in graph.contraction_steps) == (2, 2, 2, 4, 6)


def test_tensorkrowch_contracted_demo_saves_figure_without_showing() -> None:
    pytest.importorskip("tensorkrowch")
    pytest.importorskip("torch")
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_tk_contracted_save")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tensorkrowch-mps-contracted-demo.png"

    exit_code = module.main(
        [
            "tensorkrowch",
            "mps",
            "--contracted",
            "--n-sites",
            "6",
            "--save",
            str(output_path),
            "--no-show",
        ]
    )

    assert exit_code == 0
    image = assert_readable_image(output_path)
    assert image.shape[0] > 0
    assert image.shape[1] > 0


def test_run_all_examples_engines_2d_matches_new_matrix() -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_engines")

    commands = module.select_example_commands(group="engines", views="2d")
    argvs = {command.argv for command in commands}

    assert ("examples/run_demo.py", "tensorkrowch", "disconnected", "--view", "2d") in argvs
    assert ("examples/run_demo.py", "tensornetwork", "weird", "--view", "2d") in argvs
    assert ("examples/run_demo.py", "quimb", "hyper", "--view", "2d") in argvs
    assert ("examples/run_demo.py", "tenpy", "chain", "--view", "2d") in argvs
    assert ("examples/run_demo.py", "einsum", "batch", "--view", "2d") in argvs


def test_run_all_examples_themes_group_runs_overview_and_tensor_elements() -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_themes")

    commands = module.select_example_commands(group="themes", views="2d")
    argvs = {command.argv for command in commands}

    assert ("examples/run_demo.py", "themes", "overview", "--view", "2d") in argvs
    assert ("examples/run_demo.py", "themes", "tensor_elements", "--view", "2d") in argvs

    commands_3d = module.select_example_commands(group="themes", views="3d")
    argvs_3d = {command.argv for command in commands_3d}

    assert ("examples/run_demo.py", "themes", "overview", "--view", "3d") in argvs_3d
    assert ("examples/run_demo.py", "themes", "tensor_elements", "--view", "3d") not in argvs_3d


def test_themes_demo_overview_titles_all_available_themes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module(Path("examples/themes_demo.py"), "themes_demo_titles")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    args = ExampleCliArgs(
        engine="themes",
        example="overview",
        view="2d",
        labels_nodes=True,
        labels_edges=False,
        labels=None,
        hover_labels=True,
        scheme=False,
        hover_cost=False,
        tensor_inspector=False,
        contracted=False,
        from_scratch=False,
        from_list=False,
        save=None,
        no_show=False,
        n_sites=6,
        lx=3,
        ly=4,
        lz=3,
        mera_log2=3,
        tree_depth=4,
        theme="default",
    )

    fig, _ = module.run_example(args)

    assert [ax.get_title() for ax in fig.axes if ax.get_title()] == [
        "default",
        "paper",
        "colorblind",
        "dark",
        "midnight",
        "forest",
        "slate",
    ]
    fig.canvas.draw()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_themes_demo_tensor_elements_titles_all_available_themes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module(Path("examples/themes_demo.py"), "themes_demo_tensor_elements")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    args = ExampleCliArgs(
        engine="themes",
        example="tensor_elements",
        view="2d",
        labels_nodes=True,
        labels_edges=False,
        labels=None,
        hover_labels=True,
        scheme=False,
        hover_cost=False,
        tensor_inspector=False,
        contracted=False,
        from_scratch=False,
        from_list=False,
        save=None,
        no_show=False,
        n_sites=6,
        lx=3,
        ly=4,
        lz=3,
        mera_log2=3,
        tree_depth=4,
        theme="default",
    )

    fig, _ = module.run_example(args)

    assert [ax.get_title() for ax in fig.axes if ax.get_title()] == [
        "default",
        "grayscale",
        "contrast",
        "categorical",
        "paper",
        "colorblind",
        "rainbow",
        "spectral",
    ]
    fig.canvas.draw()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_themes_demo_tensor_elements_default_panel_keeps_current_default_colormap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module(
        Path("examples/themes_demo.py"),
        "themes_demo_tensor_elements_default",
    )
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    args = ExampleCliArgs(
        engine="themes",
        example="tensor_elements",
        view="2d",
        labels_nodes=True,
        labels_edges=False,
        labels=None,
        hover_labels=True,
        scheme=False,
        hover_cost=False,
        tensor_inspector=False,
        contracted=False,
        from_scratch=False,
        from_list=False,
        save=None,
        no_show=False,
        n_sites=6,
        lx=3,
        ly=4,
        lz=3,
        mera_log2=3,
        tree_depth=4,
        theme="default",
    )

    fig, _ = module.run_example(args)
    default_axis = next(ax for ax in fig.axes if ax.get_title() == "default")

    assert default_axis.images
    assert default_axis.images[0].get_cmap().name == "viridis"
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_themes_demo_tensor_elements_key_presets_use_distinct_colormaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module(
        Path("examples/themes_demo.py"),
        "themes_demo_tensor_elements_distinct",
    )
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    args = ExampleCliArgs(
        engine="themes",
        example="tensor_elements",
        view="2d",
        labels_nodes=True,
        labels_edges=False,
        labels=None,
        hover_labels=True,
        scheme=False,
        hover_cost=False,
        tensor_inspector=False,
        contracted=False,
        from_scratch=False,
        from_list=False,
        save=None,
        no_show=False,
        n_sites=6,
        lx=3,
        ly=4,
        lz=3,
        mera_log2=3,
        tree_depth=4,
        theme="default",
    )

    fig, _ = module.run_example(args)
    axis_by_title = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}

    assert axis_by_title["contrast"].images[0].get_cmap().name == "CMRmap"
    assert axis_by_title["paper"].images[0].get_cmap().name == "inferno"
    assert axis_by_title["colorblind"].images[0].get_cmap().name == "cividis"
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_themes_tensor_elements_demo_saves_figure_without_showing(tmp_path: Path) -> None:
    module = _load_example_module(
        Path("examples/run_demo.py"),
        "run_demo_themes_tensor_elements",
    )
    output_path = tmp_path / "themes-tensor-elements-demo.png"

    exit_code = module.main(
        [
            "themes",
            "tensor_elements",
            "--view",
            "2d",
            "--save",
            str(output_path),
            "--no-show",
        ]
    )

    assert exit_code == 0
    image = assert_readable_image(output_path)
    assert image.shape[0] > 0
    assert image.shape[1] > 0


def test_run_all_examples_builds_headless_subprocess_command(tmp_path: Path) -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_build")

    command = module.ExampleCommand(
        slug="quimb_hyper_2d",
        argv=("examples/run_demo.py", "quimb", "hyper", "--view", "2d"),
    )
    subprocess_command = module.build_subprocess_command(
        command,
        output_dir=tmp_path,
        python_executable="python-test",
    )

    assert subprocess_command[:6] == [
        "python-test",
        "examples/run_demo.py",
        "quimb",
        "hyper",
        "--view",
        "2d",
    ]
    assert "--no-show" in subprocess_command
    assert "--save" in subprocess_command
    assert str(tmp_path / "quimb_hyper_2d.png") in subprocess_command


def test_run_all_examples_list_mode_prints_without_running_subprocesses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_list")

    def _fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called in --list mode")

    monkeypatch.setattr(module.subprocess, "run", _fail_run)

    exit_code = module.main(["--group", "geometry", "--views", "2d", "--list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "examples/run_demo.py geometry upper_triangle2d --view 2d" in captured.out


def test_run_all_examples_placements_group_includes_input_shapes() -> None:
    module = _load_example_module(
        Path("examples/run_all_examples.py"), "run_all_examples_placements"
    )

    commands = module.select_example_commands(group="placements", views="2d")
    argvs = {command.argv for command in commands}

    assert ("examples/run_demo.py", "placements", "manual_scheme", "--view", "2d") in argvs
    assert ("examples/run_demo.py", "placements", "grid2d", "--view", "2d") in argvs


def test_geometry_decorated_sparse_grid2d_uses_flat_tensor_list() -> None:
    _require_quimb()
    module = _load_example_module(
        Path("examples/geometry_demo.py"),
        "geometry_demo_decorated_sparse_grid2d",
    )

    network = module._decorated_sparse_grid2d()

    assert isinstance(network, list)
    assert network
    assert all(not isinstance(tensor, list) for tensor in network)
    assert any("leaf_top" in str(tensor.tags) for tensor in network)
    assert any("leaf_right" in str(tensor.tags) for tensor in network)


def test_geometry_decorated_sparse_grid2d_has_expected_length_and_width() -> None:
    module = _load_example_module(
        Path("examples/geometry_demo.py"),
        "geometry_demo_decorated_sparse_grid2d_shape",
    )

    active = module._decorated_sparse_grid2d_active()
    row_widths: dict[int, int] = {}
    for row, _col in active:
        row_widths[row] = row_widths.get(row, 0) + 1

    assert max(col for _row, col in active) + 1 == 15
    assert max(row_widths.values()) == 5


def test_run_all_examples_all_group_contains_more_commands_than_default() -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_all")

    default_commands = module.select_example_commands(group="engines", views="both")
    all_commands = module.select_example_commands(group="all", views="both")

    assert len(all_commands) > len(default_commands)


def test_representative_demo_tensors_are_deterministic_and_non_constant() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tensorkrowch")
    pytest.importorskip("tensornetwork")
    pytest.importorskip("quimb.tensor")
    pytest.importorskip("tenpy")

    demo_cli = importlib.import_module("demo_cli")
    tk_demo = importlib.import_module("tensorkrowch_demo")
    tn_demo = importlib.import_module("tensornetwork_demo")
    quimb_demo = importlib.import_module("quimb_demo")
    tenpy_demo = importlib.import_module("tenpy_demo")
    einsum_demo = importlib.import_module("einsum_demo")

    args = demo_cli.ExampleCliArgs(
        engine="tensorkrowch",
        example="mps",
        view="2d",
        labels_nodes=True,
        labels_edges=False,
        labels=None,
        hover_labels=True,
        scheme=False,
        hover_cost=False,
        tensor_inspector=False,
        contracted=False,
        from_scratch=False,
        from_list=False,
        save=None,
        no_show=True,
        n_sites=3,
        lx=2,
        ly=2,
        lz=2,
        mera_log2=2,
        tree_depth=2,
    )

    tk_node_specs, tk_bond_specs = tk_demo._mps_specs(args.n_sites)
    _tk_network_a, tk_nodes_a = tk_demo._build_tensorkrowch_network(
        tk_node_specs,
        tk_bond_specs,
    )
    _tk_network_b, tk_nodes_b = tk_demo._build_tensorkrowch_network(
        tk_node_specs,
        tk_bond_specs,
    )
    tk_array_a = np.asarray(tk_nodes_a[0].tensor)
    tk_array_b = np.asarray(tk_nodes_b[0].tensor)
    assert np.array_equal(tk_array_a, tk_array_b)
    assert not np.allclose(tk_array_a, tk_array_a.flat[0])

    tn_node_specs, tn_bond_specs = tn_demo._mps_specs(args.n_sites)
    tn_nodes_a = tn_demo._build_tensornetwork_nodes(tn_node_specs, tn_bond_specs)
    tn_nodes_b = tn_demo._build_tensornetwork_nodes(tn_node_specs, tn_bond_specs)
    tn_array_a = np.asarray(tn_nodes_a[0].tensor)
    tn_array_b = np.asarray(tn_nodes_b[0].tensor)
    assert np.array_equal(tn_array_a, tn_array_b)
    assert not np.allclose(tn_array_a, tn_array_a.flat[0])

    quimb_node_specs, quimb_bond_specs = quimb_demo._mps_specs(args.n_sites)
    _quimb_network_a, quimb_tensors_a = quimb_demo._build_quimb_network(
        quimb_node_specs,
        quimb_bond_specs,
    )
    _quimb_network_b, quimb_tensors_b = quimb_demo._build_quimb_network(
        quimb_node_specs,
        quimb_bond_specs,
    )
    quimb_array_a = np.asarray(quimb_tensors_a[0].data)
    quimb_array_b = np.asarray(quimb_tensors_b[0].data)
    assert np.array_equal(quimb_array_a, quimb_array_b)
    assert not np.allclose(quimb_array_a, quimb_array_a.flat[0])

    tenpy_node_specs, tenpy_bond_specs = tenpy_demo._chain_specs(3)
    tenpy_network = tenpy_demo._build_explicit_network(tenpy_node_specs, tenpy_bond_specs)
    first_tenpy_tensor = tenpy_network.nodes[0][1]
    tenpy_array = np.asarray(first_tenpy_tensor.to_ndarray())
    assert not np.allclose(tenpy_array, tenpy_array.flat[0])

    einsum_trace_a = einsum_demo._build_mps_auto(3)
    einsum_trace_b = einsum_demo._build_mps_auto(3)
    playback_a = _extract_playback_step_records(einsum_trace_a)
    playback_b = _extract_playback_step_records(einsum_trace_b)
    assert playback_a is not None
    assert playback_b is not None
    assert playback_a[0].record is not None
    assert playback_b[0].record is not None
    einsum_array_a = np.asarray(playback_a[0].record.array)
    einsum_array_b = np.asarray(playback_b[0].record.array)
    assert np.array_equal(einsum_array_a, einsum_array_b)
    assert not np.allclose(einsum_array_a, einsum_array_a.flat[0])


def test_tensor_elements_demo_saves_figure_without_showing(tmp_path: Path) -> None:
    module = _load_example_module(Path("examples/tensor_elements_demo.py"), "tensor_elements_demo")
    output_path = tmp_path / "tensor-elements-demo.png"

    fig, _ = module.main(show=False)
    fig.savefig(output_path, bbox_inches="tight")

    image = assert_readable_image(output_path)
    assert image.shape[0] > 0


def test_tensor_elements_structured_demo_includes_sparse_and_nonfinite_cases() -> None:
    module = _load_example_module(
        Path("examples/tensor_elements_demo.py"), "tensor_elements_demo_cases"
    )

    nodes = module.build_structured_network()
    sparse_node = next(node for node in nodes if node.name == "SparseMask")
    special_node = next(node for node in nodes if node.name == "Specials")

    sparse_array = np.asarray(sparse_node.tensor)
    special_array = np.asarray(special_node.tensor)

    assert np.count_nonzero(sparse_array) < sparse_array.size // 4
    assert np.isnan(special_array).any()
    assert np.isposinf(special_array).any()
    assert np.isneginf(special_array).any()
