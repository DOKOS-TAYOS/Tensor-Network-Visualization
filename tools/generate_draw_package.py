"""Generate _core/draw/* from monolithic _draw_common.py (one slice at a time)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parent.parent
CORE = REPO / "src" / "tensor_network_viz" / "_core"
DRAW = CORE / "draw"
MONO = CORE / "_draw_common.py"


def sl(mono_lines: list[str], a: int, b: int) -> str:
    return "".join(mono_lines[a - 1 : b])


def main() -> None:
    mono_src = MONO.read_text(encoding="utf-8")
    if len(mono_src) < 5000 or "def _draw_graph" not in mono_src:
        raise SystemExit(
            f"Refusing to run: {MONO} looks like the shim ({len(mono_src)} chars). "
            "Restore the monolithic _draw_common.py from git first."
        )
    mono_lines = mono_src.splitlines(keepends=True)
    DRAW.mkdir(parents=True, exist_ok=True)

    constants_all = dedent(
        """
    __all__ = [
        "_AXIS_TIE_EPS",
        "_CURVE_NEAR_PAIR_REF",
        "_CURVE_OFFSET_FACTOR",
        "_CURVE_TANGENT_BLEND_LAMBDA",
        "_EDGE_INDEX_LABEL_ALONG_FRAC",
        "_EDGE_INDEX_LABEL_FONT_GLOBAL_SCALE",
        "_EDGE_INDEX_LABEL_GID",
        "_EDGE_INDEX_LABEL_SPAN_FRAC_CONTRACT",
        "_EDGE_INDEX_LABEL_SPAN_FRAC_PHYS",
        "_EDGE_INDEX_LABEL_WIDTH_CALIB",
        "_EDGE_INDEX_NODE_CLEAR_FRAC",
        "_EDGE_LINE_CAP_STYLE",
        "_EDGE_LINE_JOIN_STYLE",
        "_FIGURE_MIN_PX_REF",
        "_HOVER_EDGE_PICK_RADIUS_PX",
        "_INDEX_LABEL_2D_PERP_EXTRA",
        "_INDEX_LABEL_2D_STROKE_PAD",
        "_LABEL_FONT_3D_SCALE",
        "_NODE_LABEL_MARGIN_FACTOR",
        "_OCTAHEDRON_EDGE_LINEWIDTH_FACTOR",
        "_OCTAHEDRON_EDGE_LINEWIDTH_MIN",
        "_OCTAHEDRON_TRI_COUNT",
        "_PHYS_DANGLING_2D_FRAC_FROM_TIP",
        "_PHYSICAL_INDEX_LABEL_FONT_SCALE",
        "_STROKE_LABEL_EM_PERP_FRAC",
        "_STROKE_LABEL_EM_PERP_MAX_HW_MULT",
        "_STROKE_LABEL_GEOM_NORMAL_DOT_MIN",
        "_TENSOR_LABEL_GID",
        "_TENSOR_LABEL_INSIDE_FILL",
        "_TEXT_RENDER_DIAGONAL_FACTOR",
        "_UNIT_NODE_TRIS",
        "_ZOOM_FONT_CLAMP",
        "_ZORDER_EDGE_INDEX_LABEL",
        "_ZORDER_NODE_DISK",
        "_ZORDER_TENSOR_NAME",
    ]
    """
    )
    (DRAW / "constants.py").write_text(
        '"""Drawing tuning constants (shared 2D/3D)."""\n\n'
        "from __future__ import annotations\n\n"
        "import numpy as np\n\n"
        + sl(mono_lines, 44, 93)
        + "\n"
        + sl(mono_lines, 620, 637)
        + "\n"
        + sl(mono_lines, 1056, 1056)
        + "\n"
        + sl(mono_lines, 2178, 2178)
        + "\n"
        + constants_all,
        encoding="utf-8",
    )

    # fonts_and_scale.py
    (DRAW / "fonts_and_scale.py").write_text(
        "from __future__ import annotations\n\n"
        "import functools\n"
        "import math\n"
        "from collections import defaultdict\n"
        "from contextlib import suppress\n"
        "from dataclasses import dataclass\n"
        "from typing import Any, Literal, Protocol, cast\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from matplotlib.collections import LineCollection, PatchCollection\n"
        "from matplotlib.figure import Figure\n"
        "from matplotlib.font_manager import FontProperties\n"
        "from matplotlib.patches import Circle\n"
        "from matplotlib.textpath import TextPath\n"
        "from mpl_toolkits.mplot3d import proj3d\n"
        "from mpl_toolkits.mplot3d.art3d import Poly3DCollection\n\n"
        "from ...config import PlotConfig\n"
        "from .._label_format import format_tensor_node_label\n"
        "from ..contractions import _ContractionGroups, _group_contractions\n"
        "from ..curves import (\n"
        "    _ellipse_points,\n"
        "    _ellipse_points_3d,\n"
        "    _quadratic_curve,\n"
        "    _require_self_endpoints,\n"
        ")\n"
        "from ..graph import (\n"
        "    _EdgeData,\n"
        "    _EdgeEndpoint,\n"
        "    _endpoint_index_caption,\n"
        "    _GraphData,\n"
        "    _require_contraction_endpoints,\n"
        ")\n"
        "from ..layout import (\n"
        "    AxisDirections,\n"
        "    NodePositions,\n"
        "    _orthogonal_unit,\n"
        ")\n\n"
        "from .constants import *\n\n"
        + sl(mono_lines, 1265, 1278)
        + "\n"
        + sl(mono_lines, 1389, 1413)
        + "\n"
        + sl(mono_lines, 2120, 2216)
        + "\n"
        + dedent(
            """
        __all__ = [
            "_DrawScaleParams",
            "_draw_scale_params",
            "_figure_base_size_scale",
            "_figure_relative_font_scale",
            "_figure_size_sqrt_ratio",
            "_index_label_bbox_pad",
            "_on_2d_limits_changed",
            "_register_2d_zoom_font_scaling",
            "_textpath_width_pts",
        ]
        """
        ),
        encoding="utf-8",
    )

    # Fix fonts: line 2121 in mono is @dataclass - we need full class
    viewport_header = (
        "from __future__ import annotations\n\n"
        "import functools\n"
        "import math\n"
        "from collections import defaultdict\n"
        "from contextlib import suppress\n"
        "from dataclasses import dataclass\n"
        "from typing import Any, Literal, Protocol, cast\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from matplotlib.collections import LineCollection, PatchCollection\n"
        "from matplotlib.figure import Figure\n"
        "from matplotlib.font_manager import FontProperties\n"
        "from matplotlib.patches import Circle\n"
        "from matplotlib.textpath import TextPath\n"
        "from mpl_toolkits.mplot3d import proj3d\n"
        "from mpl_toolkits.mplot3d.art3d import Poly3DCollection\n\n"
        "from ...config import PlotConfig\n"
        "from .._label_format import format_tensor_node_label\n"
        "from ..contractions import _ContractionGroups, _group_contractions\n"
        "from ..curves import (\n"
        "    _ellipse_points,\n"
        "    _ellipse_points_3d,\n"
        "    _quadratic_curve,\n"
        "    _require_self_endpoints,\n"
        ")\n"
        "from ..graph import (\n"
        "    _EdgeData,\n"
        "    _EdgeEndpoint,\n"
        "    _endpoint_index_caption,\n"
        "    _GraphData,\n"
        "    _require_contraction_endpoints,\n"
        ")\n"
        "from ..layout import (\n"
        "    AxisDirections,\n"
        "    NodePositions,\n"
        "    _orthogonal_unit,\n"
        ")\n\n"
        "from .constants import *\n\n"
        "from .fonts_and_scale import _DrawScaleParams, _textpath_width_pts\n"
        "from .vectors import _perpendicular_3d\n\n"
    )

    viewport_body = (
        sl(mono_lines, 96, 618)
        + "\n"
        + sl(mono_lines, 1281, 1386)  # bond_endpoints through edge_index_fontsize
        + "\n"
    )

    viewport_all = dedent(
        """
    __all__ = [
        "_apply_axis_limits_with_outset",
        "_apply_edge_line_style",
        "_apply_text_no_clip",
        "_blend_bond_tangent_with_chord_2d",
        "_blend_bond_tangent_with_chord_3d",
        "_bond_endpoints_xyz3",
        "_bond_index_label_perp_offset",
        "_bond_reference_span_px_for_font",
        "_contraction_edge_index_label_2d_placement",
        "_contraction_edge_index_label_3d_placement",
        "_edge_index_along_bond_text_kw",
        "_edge_index_font_em_data_2d",
        "_edge_index_fontsize_for_bond",
        "_edge_index_label_axis_tie_vertical_2d",
        "_edge_index_label_is_vertical_axis_2d",
        "_edge_index_label_span_frac",
        "_edge_index_rim_arc_from_endpoint",
        "_edge_index_text_kw_tangent_stroke_align",
        "_line_halfwidth_data_2d",
        "_max_perpendicular_bond_curve_offset",
        "_nominal_figure_px_per_data_unit_3d",
        "_point_tangent_along_polyline_from_end",
        "_point_tangent_along_polyline_from_start",
        "_polyline_arc_length_total",
        "_self_loop_spatial_extent",
        "_stack_visible_tensor_coords",
        "_stroke_index_normal_screen_unit_2d",
        "_stroke_perp_distance_data_units_3d",
        "_tangent_screen_angle_deg",
        "_upright_screen_text_rotation_deg_raw",
        "_view_outset_margin_data_units",
    ]
    """
    )

    (DRAW / "viewport_geometry.py").write_text(
        viewport_header + viewport_body + viewport_all, encoding="utf-8"
    )

    vectors_body = sl(mono_lines, 835, 864)
    (DRAW / "vectors.py").write_text(
        "from __future__ import annotations\n\n"
        "from typing import Literal\n\n"
        "import numpy as np\n\n"
        + vectors_body
        + "\n"
        + dedent(
            """
        __all__ = [
            "_bond_perpendicular_unoriented",
            "_perpendicular_2d",
            "_perpendicular_3d",
        ]
        """
        ),
        encoding="utf-8",
    )

    plotter_header = (
        "from __future__ import annotations\n\n"
        "import functools\n"
        "import math\n"
        "from collections import defaultdict\n"
        "from contextlib import suppress\n"
        "from dataclasses import dataclass\n"
        "from typing import Any, Literal, Protocol, cast\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from matplotlib.collections import LineCollection, PatchCollection\n"
        "from matplotlib.figure import Figure\n"
        "from matplotlib.font_manager import FontProperties\n"
        "from matplotlib.patches import Circle\n"
        "from matplotlib.textpath import TextPath\n"
        "from mpl_toolkits.mplot3d import proj3d\n"
        "from mpl_toolkits.mplot3d.art3d import Poly3DCollection\n\n"
        "from ...config import PlotConfig\n"
        "from .._label_format import format_tensor_node_label\n"
        "from ..contractions import _ContractionGroups, _group_contractions\n"
        "from ..curves import (\n"
        "    _ellipse_points,\n"
        "    _ellipse_points_3d,\n"
        "    _quadratic_curve,\n"
        "    _require_self_endpoints,\n"
        ")\n"
        "from ..graph import (\n"
        "    _EdgeData,\n"
        "    _EdgeEndpoint,\n"
        "    _endpoint_index_caption,\n"
        "    _GraphData,\n"
        "    _require_contraction_endpoints,\n"
        ")\n"
        "from ..layout import (\n"
        "    AxisDirections,\n"
        "    NodePositions,\n"
        "    _orthogonal_unit,\n"
        ")\n\n"
        "from .constants import *\n\n"
        "from .fonts_and_scale import _DrawScaleParams\n"
        "from .viewport_geometry import (\n"
        "    _apply_axis_limits_with_outset,\n"
        "    _apply_edge_line_style,\n"
        "    _apply_text_no_clip,\n"
        ")\n\n"
    )

    (DRAW / "plotter.py").write_text(
        plotter_header
        + sl(mono_lines, 640, 832)
        + "\n"
        + dedent(
            """
        __all__ = [
            "_PlotAdapter",
            "_graph_edge_degree",
            "_make_plotter",
            "_visible_degree_one_mask",
        ]
        """
        ),
        encoding="utf-8",
    )

    labels_header = plotter_header.replace(
        "from .fonts_and_scale import _DrawScaleParams\n"
        "from .viewport_geometry import (\n"
        "    _apply_axis_limits_with_outset,\n"
        "    _apply_edge_line_style,\n"
        "    _apply_text_no_clip,\n",
        "from .fonts_and_scale import _DrawScaleParams\n",
    )
    # labels doesn't need viewport - fix labels_header to standard draw imports only
    labels_header = (
        "from __future__ import annotations\n\n"
        "import functools\n"
        "import math\n"
        "from collections import defaultdict\n"
        "from contextlib import suppress\n"
        "from dataclasses import dataclass\n"
        "from typing import Any, Literal, Protocol, cast\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from matplotlib.collections import LineCollection, PatchCollection\n"
        "from matplotlib.figure import Figure\n"
        "from matplotlib.font_manager import FontProperties\n"
        "from matplotlib.patches import Circle\n"
        "from matplotlib.textpath import TextPath\n"
        "from mpl_toolkits.mplot3d import proj3d\n"
        "from mpl_toolkits.mplot3d.art3d import Poly3DCollection\n\n"
        "from ...config import PlotConfig\n"
        "from .._label_format import format_tensor_node_label\n"
        "from ..contractions import _ContractionGroups, _group_contractions\n"
        "from ..curves import (\n"
        "    _ellipse_points,\n"
        "    _ellipse_points_3d,\n"
        "    _quadratic_curve,\n"
        "    _require_self_endpoints,\n"
        ")\n"
        "from ..graph import (\n"
        "    _EdgeData,\n"
        "    _EdgeEndpoint,\n"
        "    _endpoint_index_caption,\n"
        "    _GraphData,\n"
        "    _require_contraction_endpoints,\n"
        ")\n"
        "from ..layout import (\n"
        "    AxisDirections,\n"
        "    NodePositions,\n"
        "    _orthogonal_unit,\n"
        ")\n\n"
        "from .constants import *\n\n"
        "from .fonts_and_scale import _DrawScaleParams\n\n"
    )

    (DRAW / "labels_misc.py").write_text(
        labels_header
        + sl(mono_lines, 867, 987)
        + "\n"
        + dedent(
            """
        __all__ = [
            "_contraction_hover_label_text",
            "_curve_index_outside_disk",
            "_dangling_hover_label_text",
            "_edge_index_text_kwargs",
            "_estimate_drawn_label_count",
            "_node_label_clearance",
            "_self_loop_hover_label_text",
        ]
        """
        ),
        encoding="utf-8",
    )

    pick_header = (
        "from __future__ import annotations\n\n"
        "import math\n"
        "from typing import Any\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from mpl_toolkits.mplot3d import proj3d\n\n"
    )

    (DRAW / "pick_distance.py").write_text(
        pick_header
        + sl(mono_lines, 989, 1054)
        + "\n"
        + dedent(
            """
        __all__ = [
            "_min_sqdist_point_to_polyline_display",
            "_min_sqdist_point_to_polyline_display_3d",
            "_sqdist_point_to_segment",
        ]
        """
        ),
        encoding="utf-8",
    )

    hover_header = (
        "from __future__ import annotations\n\n"
        "import math\n"
        "from contextlib import suppress\n"
        "from typing import Any\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from matplotlib.figure import Figure\n"
        "from matplotlib.collections import PatchCollection\n"
        "from mpl_toolkits.mplot3d import proj3d\n\n"
        "from ..layout import NodePositions\n\n"
        "from .constants import *\n"
        "from .fonts_and_scale import _DrawScaleParams\n"
        "from .pick_distance import (\n"
        "    _min_sqdist_point_to_polyline_display,\n"
        "    _min_sqdist_point_to_polyline_display_3d,\n"
        ")\n"
        "from .tensors import _tensor_disk_radius_px\n\n"
    )

    (DRAW / "hover.py").write_text(
        hover_header
        + sl(mono_lines, 1059, 1262)
        + "\n"
        + dedent(
            """
        __all__ = [
            "_disconnect_tensor_network_hover",
            "_register_2d_hover_labels",
            "_register_3d_hover_labels",
        ]
        """
        ),
        encoding="utf-8",
    )

    disk_header = (
        "from __future__ import annotations\n\n"
        "import math\n"
        "from typing import Any, Literal, cast\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from mpl_toolkits.mplot3d import proj3d\n\n"
        "from .fonts_and_scale import _DrawScaleParams\n\n"
    )

    (DRAW / "disk_metrics.py").write_text(
        disk_header
        + sl(mono_lines, 1934, 1978)
        + "\n"
        + dedent(
            """
        __all__ = [
            "_display_disk_radius_px_2d",
            "_display_disk_radius_px_3d",
            "_tensor_disk_radius_px",
        ]
        """
        ),
        encoding="utf-8",
    )

    tensors_header = (
        "from __future__ import annotations\n\n"
        "import functools\n"
        "import math\n"
        "from typing import Any, Literal, cast\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from matplotlib.figure import Figure\n"
        "from matplotlib.font_manager import FontProperties\n"
        "from matplotlib.textpath import TextPath\n\n"
        "from ...config import PlotConfig\n"
        "from .._label_format import format_tensor_node_label\n"
        "from ..graph import _GraphData\n"
        "from ..layout import NodePositions\n\n"
        "from .constants import *\n"
        "from .disk_metrics import _tensor_disk_radius_px\n"
        "from .fonts_and_scale import _DrawScaleParams\n"
        "from .plotter import _PlotAdapter, _visible_degree_one_mask\n"
        "from .viewport_geometry import _stack_visible_tensor_coords\n\n"
    )

    (DRAW / "tensors.py").write_text(
        tensors_header
        + sl(mono_lines, 1981, 2118)
        + "\n"
        + dedent(
            """
        __all__ = [
            "_draw_labels",
            "_draw_nodes",
            "_refit_tensor_labels_to_disks",
            "_tensor_label_data_anchor",
            "_tensor_label_fontsize_to_fit",
            "_textpath_diagonal_points_ref10",
        ]
        """
        ),
        encoding="utf-8",
    )

    # hover imports tensors — fine: hover uses _tensor_disk_radius_px from tensors;
    # tensors does not import hover.

    edges_header = (
        "from __future__ import annotations\n\n"
        "import functools\n"
        "import math\n"
        "from collections import defaultdict\n"
        "from contextlib import suppress\n"
        "from dataclasses import dataclass\n"
        "from typing import Any, Literal, Protocol, cast\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from matplotlib.collections import LineCollection, PatchCollection\n"
        "from matplotlib.figure import Figure\n"
        "from matplotlib.font_manager import FontProperties\n"
        "from matplotlib.patches import Circle\n"
        "from matplotlib.textpath import TextPath\n"
        "from mpl_toolkits.mplot3d import proj3d\n"
        "from mpl_toolkits.mplot3d.art3d import Poly3DCollection\n\n"
        "from ...config import PlotConfig\n"
        "from .._label_format import format_tensor_node_label\n"
        "from ..contractions import _ContractionGroups, _group_contractions\n"
        "from ..curves import (\n"
        "    _ellipse_points,\n"
        "    _ellipse_points_3d,\n"
        "    _quadratic_curve,\n"
        "    _require_self_endpoints,\n"
        ")\n"
        "from ..graph import (\n"
        "    _EdgeData,\n"
        "    _EdgeEndpoint,\n"
        "    _endpoint_index_caption,\n"
        "    _GraphData,\n"
        "    _require_contraction_endpoints,\n"
        ")\n"
        "from ..layout import (\n"
        "    AxisDirections,\n"
        "    NodePositions,\n"
        "    _orthogonal_unit,\n"
        ")\n\n"
        "from .constants import *\n\n"
        "from .fonts_and_scale import _DrawScaleParams\n"
        "from .labels_misc import (\n"
        "    _contraction_hover_label_text,\n"
        "    _curve_index_outside_disk,\n"
        "    _dangling_hover_label_text,\n"
        "    _edge_index_text_kwargs,\n"
        "    _node_label_clearance,\n"
        "    _self_loop_hover_label_text,\n"
        ")\n"
        "from .plotter import _PlotAdapter\n"
        "from .vectors import _bond_perpendicular_unoriented, _perpendicular_2d\n"
        "from .viewport_geometry import (\n"
        "    _blend_bond_tangent_with_chord_2d,\n"
        "    _blend_bond_tangent_with_chord_3d,\n"
        "    _bond_index_label_perp_offset,\n"
        "    _contraction_edge_index_label_2d_placement,\n"
        "    _contraction_edge_index_label_3d_placement,\n"
        "    _edge_index_along_bond_text_kw,\n"
        "    _edge_index_fontsize_for_bond,\n"
        "    _edge_index_rim_arc_from_endpoint,\n"
        "    _point_tangent_along_polyline_from_end,\n"
        "    _point_tangent_along_polyline_from_start,\n"
        "    _polyline_arc_length_total,\n"
        ")\n\n"
    )

    (DRAW / "edges.py").write_text(
        edges_header
        + sl(mono_lines, 1416, 1931)
        + "\n"
        + dedent(
            """
        __all__ = [
            "_curved_edge_points",
            "_draw_contraction_edge",
            "_draw_dangling_edge",
            "_draw_edges",
            "_draw_self_loop_edge",
            "_plot_contraction_index_captions",
        ]
        """
        ),
        encoding="utf-8",
    )

    gp_header = (
        "from __future__ import annotations\n\n"
        "import functools\n"
        "import math\n"
        "from collections import defaultdict\n"
        "from contextlib import suppress\n"
        "from dataclasses import dataclass\n"
        "from typing import Any, Literal, Protocol, cast\n\n"
        "import numpy as np\n"
        "from matplotlib.axes import Axes\n"
        "from matplotlib.collections import LineCollection, PatchCollection\n"
        "from matplotlib.figure import Figure\n"
        "from matplotlib.font_manager import FontProperties\n"
        "from matplotlib.patches import Circle\n"
        "from matplotlib.textpath import TextPath\n"
        "from mpl_toolkits.mplot3d import proj3d\n"
        "from mpl_toolkits.mplot3d.art3d import Poly3DCollection\n\n"
        "from ...config import PlotConfig\n"
        "from .._label_format import format_tensor_node_label\n"
        "from ..contractions import _ContractionGroups, _group_contractions\n"
        "from ..curves import (\n"
        "    _ellipse_points,\n"
        "    _ellipse_points_3d,\n"
        "    _quadratic_curve,\n"
        "    _require_self_endpoints,\n"
        ")\n"
        "from ..graph import (\n"
        "    _EdgeData,\n"
        "    _EdgeEndpoint,\n"
        "    _endpoint_index_caption,\n"
        "    _GraphData,\n"
        "    _require_contraction_endpoints,\n"
        ")\n"
        "from ..layout import (\n"
        "    AxisDirections,\n"
        "    NodePositions,\n"
        "    _orthogonal_unit,\n"
        ")\n\n"
        "from .constants import *\n\n"
        "from .edges import _draw_edges\n"
        "from .fonts_and_scale import (\n"
        "    _draw_scale_params,\n"
        "    _figure_relative_font_scale,\n"
        "    _register_2d_zoom_font_scaling,\n"
        ")\n"
        "from .hover import (\n"
        "    _disconnect_tensor_network_hover,\n"
        "    _register_2d_hover_labels,\n"
        "    _register_3d_hover_labels,\n"
        ")\n"
        "from .labels_misc import _estimate_drawn_label_count\n"
        "from .plotter import _make_plotter\n"
        "from .tensors import _draw_labels, _draw_nodes, _refit_tensor_labels_to_disks\n"
        "from .viewport_geometry import (\n"
        "    _apply_axis_limits_with_outset,\n"
        "    _stack_visible_tensor_coords,\n"
        "    _view_outset_margin_data_units,\n"
        ")\n\n"
    )

    (DRAW / "graph_pipeline.py").write_text(
        gp_header + sl(mono_lines, 2219, 2328) + "\n" + '__all__ = ["_draw_graph"]\n',
        encoding="utf-8",
    )

    init_body = dedent(
        '''
    """Internal tensor-network drawing primitives (split from legacy `_draw_common`)."""

    from __future__ import annotations

    from . import (
        constants,
        disk_metrics,
        edges,
        fonts_and_scale,
        graph_pipeline,
        hover,
        labels_misc,
        pick_distance,
        plotter,
        tensors,
        vectors,
        viewport_geometry,
    )

    __all__ = [
        *constants.__all__,
        *disk_metrics.__all__,
        *edges.__all__,
        *fonts_and_scale.__all__,
        *graph_pipeline.__all__,
        *hover.__all__,
        *labels_misc.__all__,
        *pick_distance.__all__,
        *plotter.__all__,
        *tensors.__all__,
        *vectors.__all__,
        *viewport_geometry.__all__,
    ]

    globals().update({n: getattr(constants, n) for n in constants.__all__})
    globals().update({n: getattr(disk_metrics, n) for n in disk_metrics.__all__})
    globals().update({n: getattr(edges, n) for n in edges.__all__})
    globals().update({n: getattr(fonts_and_scale, n) for n in fonts_and_scale.__all__})
    globals().update({n: getattr(graph_pipeline, n) for n in graph_pipeline.__all__})
    globals().update({n: getattr(hover, n) for n in hover.__all__})
    globals().update({n: getattr(labels_misc, n) for n in labels_misc.__all__})
    globals().update({n: getattr(pick_distance, n) for n in pick_distance.__all__})
    globals().update({n: getattr(plotter, n) for n in plotter.__all__})
    globals().update({n: getattr(tensors, n) for n in tensors.__all__})
    globals().update({n: getattr(vectors, n) for n in vectors.__all__})
    globals().update({n: getattr(viewport_geometry, n) for n in viewport_geometry.__all__})
    '''
    )

    (DRAW / "__init__.py").write_text(init_body, encoding="utf-8")

    # Shim _draw_common.py
    (CORE / "_draw_common.py").write_text(
        '"""Shared scale and style parameters for 2D and 3D drawing."""\n\n'
        "from __future__ import annotations\n\n"
        "from .draw import *  # noqa: F403\n"
        "from .draw import __all__ as __all__\n",
        encoding="utf-8",
    )

    print("Generated draw/ package and _draw_common.py shim.")


if __name__ == "__main__":
    main()
