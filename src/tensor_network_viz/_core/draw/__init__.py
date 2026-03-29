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
