"""Builds a tube (generalized cylinder) mesh along the reconstructed
centerline, using the RMF frame + radius profile from geometry_engine.py.

At each of the 50 centerline samples, a ring of `sides` vertices is placed
in the plane spanned by (normal, binormal), at the sample's own radius.
Consecutive rings are connected into quads (as two triangles each). The MCP
end gets a simple flat fan cap (it's where the finger disappears into the
hand, so a flat cut reads fine); the TIP end gets a rounded hemispherical
cap built from `tip_cap_rings` extra rings shrinking toward an apex point,
so it looks like a fingertip rather than a cylinder sawn off flat. Per-vertex
normals are computed analytically (radial on the tube, blended radial+axial
on the dome) so viewers can smooth-shade the mesh instead of faceting it.
"""
import numpy as np

DEFAULT_TIP_CAP_RINGS = 6


def build_finger_mesh(geometry, sides=24, tip_cap_rings=DEFAULT_TIP_CAP_RINGS):
    """geometry: the dict from geometry_engine.reconstruct_full_geometry.
    Returns (vertices, faces, normals): vertices (N,3) float, faces (M,3) int
    (0-indexed, into vertices), normals (N,3) unit float, all lengths in
    millimeters.
    """
    centerline = geometry["centerline"]
    tangent = geometry["tangent"]
    normal = geometry["normal"]
    binormal = geometry["binormal"]
    radius = geometry["radius"]
    n_rings = len(centerline)

    theta = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # tube rings: vertex = centerline[i] + radius[i] * radial_dir, normal = radial_dir
    tube_vertices = np.empty((n_rings, sides, 3), dtype=np.float64)
    tube_normals = np.empty((n_rings, sides, 3), dtype=np.float64)
    for i in range(n_rings):
        radial_dir = np.outer(cos_t, normal[i]) + np.outer(sin_t, binormal[i])
        tube_vertices[i] = centerline[i] + radius[i] * radial_dir
        tube_normals[i] = radial_dir

    # rounded tip cap: rings at polar angle phi in (0, pi/2) from the tip
    # center, radius shrinking as cos(phi) and pushed out along the tangent
    # by sin(phi) - a hemisphere of the same radius as the last tube ring,
    # converging on an apex one radius further out.
    tip_center = centerline[-1]
    r_tip = radius[-1]
    t_tip, n_tip, b_tip = tangent[-1], normal[-1], binormal[-1]
    radial_dir_tip = np.outer(cos_t, n_tip) + np.outer(sin_t, b_tip)

    phis = np.linspace(0.0, np.pi / 2, tip_cap_rings + 2)[1:-1]
    cap_vertices = np.empty((tip_cap_rings, sides, 3), dtype=np.float64)
    cap_normals = np.empty((tip_cap_rings, sides, 3), dtype=np.float64)
    for j, phi in enumerate(phis):
        cap_vertices[j] = (tip_center + r_tip * np.sin(phi) * t_tip
                            + r_tip * np.cos(phi) * radial_dir_tip)
        cap_normals[j] = np.cos(phi) * radial_dir_tip + np.sin(phi) * t_tip

    apex = tip_center + r_tip * t_tip
    apex_normal = t_tip

    mcp_center = centerline[0]
    mcp_normal = -tangent[0]

    all_vertices = np.concatenate([
        tube_vertices.reshape(-1, 3),
        cap_vertices.reshape(-1, 3),
        mcp_center[None, :],
        apex[None, :],
    ])
    all_normals = np.concatenate([
        tube_normals.reshape(-1, 3),
        cap_normals.reshape(-1, 3),
        mcp_normal[None, :],
        apex_normal[None, :],
    ])

    n_ring_rows = n_rings + tip_cap_rings  # tube rings + cap rings, ring-indexed the same way
    mcp_center_idx = n_ring_rows * sides
    apex_idx = mcp_center_idx + 1

    def ring_idx(ring, k):
        return ring * sides + (k % sides)

    faces = []
    for i in range(n_ring_rows - 1):
        for k in range(sides):
            a, b = ring_idx(i, k), ring_idx(i, k + 1)
            c, d = ring_idx(i + 1, k), ring_idx(i + 1, k + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])

    for k in range(sides):
        faces.append([mcp_center_idx, ring_idx(0, k + 1), ring_idx(0, k)])
    for k in range(sides):
        faces.append([apex_idx, ring_idx(n_ring_rows - 1, k), ring_idx(n_ring_rows - 1, k + 1)])

    return all_vertices, np.asarray(faces, dtype=np.int64), all_normals


def save_obj(path, vertices, faces, normals=None):
    """Minimal Wavefront OBJ writer - opens in Blender/MeshLab/most 3D
    viewers for a real look at the mesh outside the live cv2 window.
    Writes per-vertex normals (if given) so viewers smooth-shade the mesh
    instead of faceting it at every ring."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        if normals is not None:
            for vn in normals:
                f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
            for tri in faces:
                idx = [i + 1 for i in tri]  # OBJ is 1-indexed
                f.write(f"f {idx[0]}//{idx[0]} {idx[1]}//{idx[1]} {idx[2]}//{idx[2]}\n")
        else:
            for tri in faces:
                f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
