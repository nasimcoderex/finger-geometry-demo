"""The "GLB Ring" pipeline stage: procedurally builds a ring band mesh (with
a flower ornament on its front face) and exports/imports it as a real binary
glTF (.glb) file via trimesh, so the asset genuinely exists on disk as a GLB
- not just an in-memory array - even though the current renderer (ring.py)
is Python/OpenCV rather than Three.js.

Canonical local space: the ring's hole axis is the local Z axis, the band is
centered at the origin, and INNER_RADIUS (the ring's inner contact radius,
matching how real rings are sized) is fixed at 1.0 unit. Callers scale this
to real pixel size at render time by multiplying by the finger's metric
radius (mm, from the Finger Geometry Network) converted to pixels.

The flower sits at local theta=90 deg on the band's outer surface - not an
arbitrary choice: ring.py's render frame maps local X -> "across the visible
finger" and local Y -> its own synthetic depth axis (toward camera), so the
point where local Y is the outward radial direction (theta=90, radial_dir =
(0,1,0)) is always the camera-facing point of the band, for any hand pose.
Baking the flower there means it always renders front-and-center with no
extra per-frame logic.
"""
from pathlib import Path

import numpy as np

INNER_RADIUS = 1.0        # canonical unit - callers scale this to real size
BAND_THICKNESS = 0.22     # radial thickness of the metal, in units of INNER_RADIUS
BAND_WIDTH = 0.55         # extent along the hole axis (how "tall" the band is)
# Higher than the original 48x16: flat per-triangle shading (no true Gouraud/
# Phong interpolation in the cv2 rasterizer) reads as smooth roughly in
# proportion to how small each facet is on screen - doubling resolution here
# is the cheapest lever toward "polished metal" instead of "faceted toy".
THETA_SEGMENTS = 96       # around the finger
PHI_SEGMENTS = 28         # around the band's own cross-section

GOLD_RGBA = (0.90, 0.70, 0.20, 1.0)    # richer/more saturated than before, closer to real gold
PETAL_RGBA = (0.86, 0.42, 0.55, 1.0)   # soft rose/enamel tone, contrasts with the gold band
CENTER_RGBA = (0.95, 0.86, 0.55, 1.0)  # warm gold-white "pistil"
DIAMOND_RGBA = (0.94, 0.97, 1.0, 1.0)  # icy white-blue - flat per-facet shading gives this its sparkle

FLOWER_N_PETALS = 6
FLOWER_PETAL_OFFSET = 0.30   # local units, distance from flower center to each petal
FLOWER_CENTER_DOME = dict(radius_out=0.15, radius_side=0.15, height=0.11, segments_u=20, segments_v=8)
FLOWER_PETAL_DOME = dict(radius_out=0.24, radius_side=0.13, height=0.085, segments_u=20, segments_v=8)

# Brilliant-cut-ish gem: girdle (widest ring) -> crown (sloped facets up to a
# flat table) and -> pavilion (sloped facets down to a point, the culet).
# Real diamonds have a deeper pavilion than crown, hence the height split.
GEM_FACETS = 8
GEM_GIRDLE_RADIUS = 0.24
GEM_TABLE_RADIUS = 0.12
GEM_CROWN_HEIGHT = 0.11
GEM_PAVILION_HEIGHT = 0.20
GEM_PRONG_COUNT = 4
GEM_PRONG_DOME = dict(radius_out=0.045, radius_side=0.045, height=0.09, segments_u=8, segments_v=5)

# --- real asset support (user-supplied diamondring.glb) ---------------------
REAL_ASSET_GLB_PATH = Path(__file__).parent / "models" / "diamondring.glb"
REAL_ASSET_CACHE_PATH = Path(__file__).parent / "models" / "diamondring_decimated.glb"
REAL_ASSET_BAND_KEY = "dmesh_1"    # names inside diamondring.glb's Scene, found by inspection
REAL_ASSET_GEM_KEY = "dmesh"
# 1.4k faces produced spiky/jagged artifacts in the prong+basket setting (a
# complex shape relative to its size) - render-time headroom is large (only
# ~11ms/frame at 2.5k total faces), so keep far more detail than strictly
# necessary for performance.
REAL_ASSET_BAND_FACE_TARGETS = (15000, 8000, 5000)  # multi-pass - one pass alone plateaus early
REAL_ASSET_GEM_FACE_TARGETS = (1200,)
REAL_ASSET_BAND_RGBA = (0.80, 0.80, 0.82, 1.0)   # from the glTF material's baseColorFactor (204,204,204)
REAL_ASSET_GEM_RGBA = DIAMOND_RGBA               # asset has no baseColorFactor/texture -> glTF default white

DEFAULT_GLB_PATH = Path(__file__).parent / "models" / "dummy_ring.glb"


def _dome(center, right, up, forward, radius_out, radius_side, height,
          segments_u=12, segments_v=5):
    """A quarter-ellipsoid bump (flat base, rounded top) sitting on a
    surface: `up` is the direction it rises along, `right`/`forward` are the
    two in-surface axes (radius_out along `right`, radius_side along
    `forward` - letting petals be elongated ellipses rather than circles).
    Returns (vertices, faces, normals) in the same space as center/right/up/
    forward are expressed in (here: the ring's canonical local space)."""
    theta = np.linspace(0.0, 2 * np.pi, segments_u, endpoint=False)
    phi = np.linspace(0.0, np.pi / 2, segments_v + 1)  # 0=base rim, pi/2=pole
    cos_th, sin_th = np.cos(theta), np.sin(theta)

    verts = np.empty((len(phi), segments_u, 3))
    norms = np.empty((len(phi), segments_u, 3))
    for iv, ph in enumerate(phi):
        cos_ph, sin_ph = np.cos(ph), np.sin(ph)
        x = cos_th * cos_ph * radius_out
        y = np.full_like(cos_th, sin_ph * height)
        z = sin_th * cos_ph * radius_side
        verts[iv] = center + np.outer(x, right) + np.outer(y, up) + np.outer(z, forward)

        nx = x / max(radius_out, 1e-6) ** 2
        ny = y / max(height, 1e-6) ** 2
        nz = z / max(radius_side, 1e-6) ** 2
        n_world = np.outer(nx, right) + np.outer(ny, up) + np.outer(nz, forward)
        norms[iv] = n_world / np.clip(np.linalg.norm(n_world, axis=1, keepdims=True), 1e-9, None)

    def vid(iv, iu):
        return iv * segments_u + (iu % segments_u)

    faces = []
    for iv in range(len(phi) - 1):
        for iu in range(segments_u):
            a, b, c, d = vid(iv, iu), vid(iv, iu + 1), vid(iv + 1, iu + 1), vid(iv + 1, iu)
            faces.append([a, b, c])
            faces.append([a, c, d])

    return verts.reshape(-1, 3), np.asarray(faces, dtype=np.int64), norms.reshape(-1, 3)


def _build_flower(outer_radius, n_petals=FLOWER_N_PETALS, petal_offset=FLOWER_PETAL_OFFSET,
                   center_dome=FLOWER_CENTER_DOME, petal_dome=FLOWER_PETAL_DOME):
    """Builds the flower ornament at local theta=90 (outward = local Y).
    axis_a (local X) is the band's circumference direction there, axis_b
    (local Z) is the band-width direction - both flat, valid in-surface
    petal-layout axes for a small motif like this."""
    flower_center = np.array([0.0, outer_radius, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    axis_a = np.array([1.0, 0.0, 0.0])
    axis_b = np.array([0.0, 0.0, 1.0])

    all_verts, all_faces, all_norms, all_colors = [], [], [], []

    def add_part(verts, faces, norms, color_rgba):
        offset = sum(len(v) for v in all_verts)
        all_verts.append(verts)
        all_faces.append(faces + offset)
        all_norms.append(norms)
        all_colors.append(np.tile(color_rgba, (len(verts), 1)))

    cv, cf, cn = _dome(flower_center, axis_a, up, axis_b, **center_dome)
    add_part(cv, cf, cn, CENTER_RGBA)

    for k in range(n_petals):
        ang = 2 * np.pi * k / n_petals
        petal_dir = np.cos(ang) * axis_a + np.sin(ang) * axis_b
        petal_side = np.cross(up, petal_dir)
        petal_center = flower_center + petal_dir * petal_offset
        pv, pf, pn = _dome(petal_center, petal_dir, up, petal_side, **petal_dome)
        add_part(pv, pf, pn, PETAL_RGBA)

    return (np.concatenate(all_verts), np.concatenate(all_faces),
            np.concatenate(all_norms), np.concatenate(all_colors))


def _outward_normal(a, b, c, interior_ref):
    """Flat face normal for triangle (a,b,c), sign-corrected to point away
    from `interior_ref` (a point inside the gem's rough volume) - simpler
    and more robust than reasoning about winding order by hand for each of
    the three facet families (pavilion/crown/table) below."""
    n = np.cross(b - a, c - a)
    n = n / np.clip(np.linalg.norm(n), 1e-9, None)
    if np.dot(n, ((a + b + c) / 3.0) - interior_ref) < 0:
        n = -n
    return n


def _build_diamond_setting(outer_radius, facets=GEM_FACETS, girdle_radius=GEM_GIRDLE_RADIUS,
                            table_radius=GEM_TABLE_RADIUS, crown_height=GEM_CROWN_HEIGHT,
                            pavilion_height=GEM_PAVILION_HEIGHT, prong_count=GEM_PRONG_COUNT,
                            prong_dome=GEM_PRONG_DOME):
    """A faceted gem mounted on the band's outer surface at local theta=90
    (see module docstring for why that angle is always camera-facing),
    culet resting near the surface (h=0), girdle at h=pavilion_height, table
    at h=pavilion_height+crown_height. Each facet gets its own duplicated
    vertices and flat normal (no smoothing across facets) - real gems are
    sharp-edged, and flat shading is what makes each facet catch light
    differently, which is what reads as "sparkle" with this renderer's
    per-triangle shading. A handful of small gold prong domes grip the
    girdle, gold to match the band."""
    center = np.array([0.0, outer_radius, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    axis_a = np.array([1.0, 0.0, 0.0])
    axis_b = np.array([0.0, 0.0, 1.0])
    interior_ref = center + up * (pavilion_height * 0.4 + 0.01)

    theta = np.linspace(0.0, 2 * np.pi, facets, endpoint=False)

    def ring_pt(radius, height, th):
        return center + up * height + axis_a * (np.cos(th) * radius) + axis_b * (np.sin(th) * radius)

    culet = center
    girdle = [ring_pt(girdle_radius, pavilion_height, th) for th in theta]
    table = [ring_pt(table_radius, pavilion_height + crown_height, th) for th in theta]
    table_center = center + up * (pavilion_height + crown_height)

    verts, faces, norms = [], [], []

    def add_tri(a, b, c):
        n = _outward_normal(a, b, c, interior_ref)
        idx = len(verts)
        verts.extend([a, b, c])
        norms.extend([n, n, n])
        faces.append([idx, idx + 1, idx + 2])

    for i in range(facets):
        i2 = (i + 1) % facets
        add_tri(culet, girdle[i], girdle[i2])                    # pavilion facet
        add_tri(girdle[i], table[i], table[i2])                  # crown facet (part 1)
        add_tri(girdle[i], table[i2], girdle[i2])                # crown facet (part 2)
        add_tri(table_center, table[i], table[i2])               # table facet

    gem_verts = np.array(verts)
    gem_faces = np.array(faces, dtype=np.int64)
    gem_norms = np.array(norms)
    gem_colors = np.tile(DIAMOND_RGBA, (len(gem_verts), 1))

    all_verts, all_faces, all_norms, all_colors = [gem_verts], [gem_faces], [gem_norms], [gem_colors]

    def add_part(v, f, n, color_rgba):
        offset = sum(len(x) for x in all_verts)
        all_verts.append(v)
        all_faces.append(f + offset)
        all_norms.append(n)
        all_colors.append(np.tile(color_rgba, (len(v), 1)))

    for k in range(prong_count):
        ang = 2 * np.pi * (k + 0.5) / prong_count  # offset from facet seams, not aligned to them
        prong_dir = np.cos(ang) * axis_a + np.sin(ang) * axis_b
        prong_side = np.cross(up, prong_dir)
        prong_center = center + prong_dir * girdle_radius * 0.85 + up * (pavilion_height - 0.02)
        pv, pf, pn = _dome(prong_center, prong_dir, up, prong_side, **prong_dome)
        add_part(pv, pf, pn, GOLD_RGBA)

    return (np.concatenate(all_verts), np.concatenate(all_faces),
            np.concatenate(all_norms), np.concatenate(all_colors))


def build_ring_mesh(inner_radius=INNER_RADIUS, band_thickness=BAND_THICKNESS,
                     band_width=BAND_WIDTH, theta_segments=THETA_SEGMENTS,
                     phi_segments=PHI_SEGMENTS, ornament="diamond"):
    """Torus band: local Z is the hole axis. Returns (vertices, faces,
    normals, colors) - vertices/normals/colors (N,3)/(N,3)/(N,4), faces
    (M,3) int, all 0-indexed."""
    major_radius = inner_radius + band_thickness / 2.0

    theta = np.linspace(0.0, 2 * np.pi, theta_segments, endpoint=False)
    phi = np.linspace(0.0, 2 * np.pi, phi_segments, endpoint=False)
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    cos_ph, sin_ph = np.cos(phi), np.sin(phi)

    radial_dir = np.stack([cos_th, sin_th, np.zeros_like(cos_th)], axis=1)  # (theta_segments,3)

    radial_mag = major_radius + (band_thickness / 2.0) * cos_ph  # (phi_segments,)
    axial_off = (band_width / 2.0) * sin_ph                       # (phi_segments,)

    vertices = (radial_dir[:, None, :] * radial_mag[None, :, None]
                + np.array([0.0, 0.0, 1.0])[None, None, :] * axial_off[None, :, None])

    hole_axis = np.array([0.0, 0.0, 1.0])
    normal_w_radial = band_width * cos_ph
    normal_w_axial = band_thickness * sin_ph
    normals_unnorm = (radial_dir[:, None, :] * normal_w_radial[None, :, None]
                       + hole_axis[None, None, :] * normal_w_axial[None, :, None])
    norms = np.linalg.norm(normals_unnorm, axis=2, keepdims=True)
    normals = normals_unnorm / np.clip(norms, 1e-9, None)

    def vid(i, j):
        return (i % theta_segments) * phi_segments + (j % phi_segments)

    faces = []
    for i in range(theta_segments):
        for j in range(phi_segments):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            faces.append([a, b, c])
            faces.append([a, c, d])

    band_vertices = vertices.reshape(-1, 3)
    band_faces = np.asarray(faces, dtype=np.int64)
    band_normals = normals.reshape(-1, 3)
    band_colors = np.tile(GOLD_RGBA, (len(band_vertices), 1))

    if ornament is None:
        return band_vertices, band_faces, band_normals, band_colors

    outer_radius = inner_radius + band_thickness
    if ornament == "diamond":
        ov, of, on, oc = _build_diamond_setting(outer_radius)
    elif ornament == "flower":
        ov, of, on, oc = _build_flower(outer_radius)
    else:
        raise ValueError(f"unknown ornament {ornament!r} (expected 'diamond', 'flower', or None)")

    all_vertices = np.concatenate([band_vertices, ov])
    all_faces = np.concatenate([band_faces, of + len(band_vertices)])
    all_normals = np.concatenate([band_normals, on])
    all_colors = np.concatenate([band_colors, oc])
    return all_vertices, all_faces, all_normals, all_colors


def _decimate_multi_pass(mesh, face_targets):
    """A single simplify_quadric_decimation call plateaus well above the
    requested face_count on this asset (verified empirically - 1.07M faces
    wouldn't go below ~8-10k in one pass regardless of target/aggression).
    Multiple passes, each decimating the previous pass's output, get much
    further (1.07M -> ~1.4k here) - each pass's local optimum becomes a
    fresh starting point for the next."""
    for target in face_targets:
        mesh = mesh.simplify_quadric_decimation(face_count=target, aggression=8)
    return mesh


def _find_hole_center_and_radius(vertices):
    """The real asset is an asymmetric solitaire design (plain circular
    shank + a raised head/setting bulging toward the gem), so the hole
    center isn't the bounding-box center. Found instead by searching for
    the (cx, cy) that makes the inner boundary of the vertex cloud most
    circular - the sweep with the most vertices packed into a thin radial
    band around some radius is the actual hole. cx uses the median (X is
    only roughly, not exactly, centered once the asset's own scene-graph
    transform is applied - see load_real_diamondring). Assumes the hole
    lies in the local XY plane (Z the hole axis) - verified empirically:
    of the three candidate planes, XY has by far the most vertices packed
    into a tight radial band (~27%) vs XZ/YZ (~13%/~22%)."""
    cx = float(np.median(vertices[:, 0]))
    best_cy, best_radius, best_score = 0.0, 1.0, -1
    for cy in np.linspace(vertices[:, 1].min(), vertices[:, 1].max(), 200):
        d = np.linalg.norm(vertices[:, :2] - [cx, cy], axis=1)
        inner_r = np.percentile(d, 8)
        score = int(((d >= inner_r) & (d <= inner_r * 1.12)).sum())
        if score > best_score:
            best_cy, best_radius, best_score = cy, inner_r, score
    return cx, best_cy, best_radius


def load_real_diamondring(asset_path=REAL_ASSET_GLB_PATH, cache_path=REAL_ASSET_CACHE_PATH,
                           force_rebuild=False):
    """Loads the user-supplied diamondring.glb, decimates it from ~1.07M
    triangles down to a few thousand (the CPU triangle-loop renderer in
    ring.py can't handle the original density), recenters/rescales it into
    this module's canonical space (hole axis = local Z, inner radius = 1.0),
    and bakes flat per-part colors (the asset uses PBR textures/materials,
    which this renderer doesn't sample - band reads its material's
    baseColorFactor, gem has neither texture nor baseColorFactor set so it
    defaults to glTF's white). Caches the processed result to `cache_path`
    since decimation takes about a second - not per-frame cost, but not
    worth repeating every launch either."""
    cache_path = Path(cache_path)
    if not force_rebuild and cache_path.exists():
        return load_glb(cache_path)

    import trimesh
    scene = trimesh.load(str(asset_path), process=False)

    # glTF node names are regenerated (non-deterministic) on every load, so
    # they can't be hardcoded - map geometry name -> its node's transform
    # fresh each time. Skipping this transform (an earlier bug) leaves both
    # parts internally consistent with each other but in the wrong overall
    # orientation/scale, which showed up as a broken/fragmented-looking mesh.
    node_transform = {}
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        node_transform[geom_name] = transform

    band = scene.geometry[REAL_ASSET_BAND_KEY].copy()
    gem = scene.geometry[REAL_ASSET_GEM_KEY].copy()
    band.apply_transform(node_transform[REAL_ASSET_BAND_KEY])
    gem.apply_transform(node_transform[REAL_ASSET_GEM_KEY])

    # Hole center/radius found on the FULL-resolution band (543k vertices,
    # post-transform, pre-decimation) - decimation moves vertex positions
    # around (that's the whole point of quadric error minimization), so
    # searching on the already-decimated ~5k-vertex mesh means the "inner
    # radius" boundary is built from noisier, shifted data. Using the full
    # mesh here costs nothing extra (same load already done above) and
    # directly fixes the ring rendering visibly off-center from the tracked
    # finger position in the live app.
    cx, cy, inner_radius = _find_hole_center_and_radius(band.vertices)
    center = np.array([cx, cy, 0.0])

    band_simple = _decimate_multi_pass(band, REAL_ASSET_BAND_FACE_TARGETS)
    gem_simple = _decimate_multi_pass(gem, REAL_ASSET_GEM_FACE_TARGETS)

    band_v = (band_simple.vertices - center) / inner_radius
    gem_v = (gem_simple.vertices - center) / inner_radius

    band_n = np.asarray(band_simple.vertex_normals)
    gem_n = np.asarray(gem_simple.vertex_normals)

    band_c = np.tile(REAL_ASSET_BAND_RGBA, (len(band_v), 1))
    gem_c = np.tile(REAL_ASSET_GEM_RGBA, (len(gem_v), 1))

    all_vertices = np.concatenate([band_v, gem_v])
    all_faces = np.concatenate([band_simple.faces, np.asarray(gem_simple.faces) + len(band_v)])
    all_normals = np.concatenate([band_n, gem_n])
    all_colors = np.concatenate([band_c, gem_c])

    export_glb(cache_path, all_vertices, all_faces, all_normals, all_colors)
    return all_vertices, all_faces, all_normals, all_colors


def export_glb(path, vertices, faces, normals, colors):
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)
    rgba_u8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba_u8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def load_glb(path):
    import trimesh
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float64) / 255.0
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.vertex_normals), colors


def ensure_ring_glb(path=DEFAULT_GLB_PATH, force_rebuild=False):
    """Builds and exports the dummy ring GLB if it doesn't exist yet, then
    loads it back - so the render stage always consumes an actual .glb file
    on disk, matching the Camera->...->GLB Ring pipeline shape."""
    path = Path(path)
    if force_rebuild or not path.exists():
        vertices, faces, normals, colors = build_ring_mesh()
        export_glb(path, vertices, faces, normals, colors)
    return load_glb(path)
