# Layout Algorithms

This page describes the internal layout rules used to place tensor nodes and free
edges. The wording is intentionally practical: it explains the choices the code
makes, not every implementation detail.

## Node Placement in 2D

The renderer first converts the input network into an internal graph. Contraction
edges connect tensors; virtual nodes represent shared hyperedges when one index
touches more than two tensors.

For each connected component, the layout code detects a simple structure when it
can:

- chains are placed on one straight line,
- rectangular 2D grids use their grid coordinates,
- rectangular 3D grids are projected into 2D with a stable oblique depth offset,
- trees use a layered tree layout,
- planar graphs use NetworkX planar placement,
- everything else falls back to a force-directed layout.

After this base placement, virtual hubs are snapped to the barycenter of their
neighbors, colocated hubs are spread apart, special one-neighbor hubs are nudged
away from the tensor disk, and trimmed leaf tensors are reattached near their
parent using local collision checks. Components are then packed next to each
other and normalized to a stable drawing scale.

## Node Placement in 3D

The 3D layout starts from the same component analysis as 2D. Regular 3D grids use
their true `(i, j, k)` coordinates. Other components are first placed in 2D and
then lifted into the `z = 0` plane.

When the lifted graph would hide important geometry, the code promotes selected
nodes onto extra z-layers. It does this for virtual hubs that overlap visible
geometry and for non-incident bonds whose 2D segments cross. This keeps simple
planar inputs planar while giving ambiguous 3D views enough depth to read.

Trimmed leaf tensors in 3D are reattached along local orthogonal directions so
they do not collapse onto their parent.

## Free Edges in 2D

Free edges are open tensor indices, sometimes called dangling edges. In 2D, each
node receives an ordered list of candidate directions. The order depends on the
component shape:

- chains prefer a common perpendicular direction first, then its opposite, then
  the chain directions,
- 2D grids process outer shells first and try to point boundary indices outward,
- projected 3D grids reuse the same 2D shell idea on the projected shape,
- trees and irregular components use stable structure-specific fallbacks.

For each free axis, the algorithm tries candidates in order and keeps the first
one that passes the local checks. A candidate is rejected when it is too close to
an already assigned axis on the same node or when the configured local 2D
neighbor checks show that the stub would enter nearby node space or cross nearby
bond/stub geometry. Random fallback directions are generated only if all
deterministic candidates fail.

Directional axis names such as `left`, `right`, `up`, `down`, `xp`, `ym`,
`north`, and `south` can force or strongly guide a free axis direction.

## Free Edges in 3D

The 3D free-edge algorithm mirrors the 2D strategy: it tries directions in a
fixed order, accepts the first valid one, and avoids generating random
directions unless every deterministic candidate has been exhausted.

Every node gets a local frame:

- `front` is the main reference direction,
- `right` and `up` complete the local orientation,
- all candidate directions are expressed in that frame.

The deterministic direction set is the 26-neighbor cube: 6 axis directions, 12
face diagonals, and 8 cube diagonals. If a node has several free edges, directions
already tried for that node are skipped for later free edges. If all 26
deterministic directions fail, the code generates 16 random unit directions. If
those also fail, the last tried direction is used.

3D conflict checks are intentionally local to the node. A candidate conflicts
only with directions already assigned to the same node, including incident bonds
and earlier free edges. It is rejected when it points in the same direction or
within 10 degrees of another same-node direction. Opposite directions are
allowed.

Structure-specific references choose the candidate order:

- chains use a common perpendicular `front`; interior chain nodes skip exact
  `right` and `left` because those directions are occupied by neighboring
  tensors,
- single surfaces start with the two normals to the surface, then surface
  directions, then diagonals that leave the plane most clearly,
- stepped surfaces use the normal of the first layer as the base reference and
  prioritize directions that leave the stack,
- 3D grids process outer shells before inner shells; face nodes act like surface
  nodes, edge nodes act like linear nodes with an outward diagonal `front`, and
  corner nodes start from the center-to-corner outward direction.

Named free axes keep their directional behavior in 3D as well. Names such as
`front`, `back`, `left`, `right`, `up`, `down`, `xp`, `xm`, `yp`, `ym`, `zp`,
and `zm` are resolved before the ordered fallback candidates.
