// Three.js ring renderer. This is the part that actually upgrades over the
// Python version's hand-rolled CPU triangle painter (ring.py's
// render_ring_overlay): real WebGL rendering of the FULL-resolution
// diamondring.glb (~1M triangles, real PBR materials) instead of a
// decimated, flat-colored CPU rasterization.
//
// Coordinate convention (must match ringPose.js exactly): an orthographic
// camera sized 1:1 to the video's pixel dimensions, with world X/Y in pixel
// units (Y increasing downward, matching image coordinates) and world Z as
// the same synthetic depth axis ringPose.js's cameraSpaceFrame defines -
// "+Z = toward camera" - reproducing the weak-perspective assumption the
// entire Python pipeline was built on (depth is used for occlusion only,
// never for size/position - exactly what orthographic projection gives for
// free, with no re-derivation needed).
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { JewelryEnvironment } from "./JewelryEnvironment.js";

const CAMERA_Z = 10000; // arbitrary large distance - only near/far range matters
const CAMERA_NEAR = 1;
const CAMERA_FAR = 20000;

// Exposure is driven per-frame by the actual video's brightness (see
// updateExposureFromLuminance) - real product photography exposure is
// matched to the scene it's shot in, and a ring rendered at a fixed
// brightness regardless of room lighting is a strong "pasted on" tell.
// These bound how far that adjustment is allowed to swing.
const EXPOSURE_MIN = 0.5;
const EXPOSURE_MAX = 1.8;
const EXPOSURE_REFERENCE_LUMINANCE = 110; // 0-255 gray value treated as "normal" room lighting
const EXPOSURE_SMOOTHING = 0.08; // low-pass so exposure doesn't flicker frame to frame

// The raw diamondring.glb is NOT in this app's canonical unit convention
// (hole centered at local origin, inner radius = 1.0) - it's whatever scale
// the original 3D asset was authored in. The Python port had to find this
// by searching for the (x,y) center that makes the band's inner surface
// most circular (ring_model.py's _find_hole_center_and_radius); these are
// the exact values that function returns for this asset (computed once via
// that existing Python logic, not re-derived here - see the conversation
// for how they were extracted). Note this is only the recenter/rescale
// step - unlike the Python port, no per-node scene-graph transform
// application is needed here, since GLTFLoader already builds the full
// scene graph with all node transforms applied automatically.
const HOLE_CENTER_X = 0.15416206511340716;
const HOLE_CENTER_Y = 0.14046122919460458;
const HOLE_INNER_RADIUS = 0.7540633272514107;

function createSoftShadowTexture(size = 128) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(0,0,0,1)");
  gradient.addColorStop(0.6, "rgba(0,0,0,0.5)");
  gradient.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export class RingRenderer {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    this.renderer.localClippingEnabled = true; // required for material.clippingPlanes to take effect
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    // ACES Filmic is what makes toneMappingExposure below actually do
    // something perceptually useful (the default NoToneMapping just clips) -
    // it's also generally what makes PBR metal/gem highlights look like
    // photography instead of a flat-clipped render.
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this._smoothedExposure = 1.0;

    this.scene = new THREE.Scene();
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
    dirLight.position.set(0, 0, 1); // camera-collocated, matches ring.py's shading convention
    this.scene.add(dirLight);

    // The band/gem materials are near-zero roughness, high metalness (the
    // Python side inspected the glTF material: roughness=0.05, metallic
    // unset -> glTF default 1.0). A metalness~1 PBR material renders almost
    // black under direct lighting alone - it needs something to reflect.
    // JewelryEnvironment is a cheap stand-in for a real HDRI (same
    // PMREMGenerator.fromScene technique Three.js's own RoomEnvironment
    // uses), lit like an actual jewelry-photography 3-light softbox rig
    // instead of a generic room - see that file for why.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new JewelryEnvironment()).texture;
    pmrem.dispose();

    this.camera = null; // built in resize(), once we know the video's pixel dimensions
    this.ringRoot = null; // Object3D wrapping the loaded GLTF scene
    this._shadowMesh = null; // soft contact shadow, child of ringRoot - see loadRingAsset
    // FIXED in world/camera space - normal.dot(p) + constant >= 0 keeps
    // world Z >= 0, i.e. "+Z = toward camera" (this module's convention,
    // matching ring.py's `view_dir = (0,0,1)` / `world[:,2] > 0` cull test
    // exactly). This must NOT be re-derived from ringRoot's own rotating
    // matrixWorld each frame (an earlier version of this code did that,
    // treating it as a plane rigidly attached to the ring's local space) -
    // that makes the test invariant to the ring's own rotation, since a
    // plane transformed by the same matrix as the points it tests keeps
    // testing the identical LOCAL half every frame regardless of hand
    // orientation. Confirmed live: with the real diamondring.glb, the gem's
    // geometry sits entirely on one side of local Y=0 (never straddles it -
    // verified against ring_model.py's own canonical-space normalization),
    // so a local-space-locked test never culls it at all, and the
    // "flip to the hidden side" hand orientations that should hide the gem
    // (see ringPose.js's palmFacingSign) render debris from mismatched
    // front-face culling on unclipped geometry instead - a dark jagged
    // shard - rather than either the correct full gem or nothing.
    this._worldClipPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

    this._width = 0;
    this._height = 0;
  }

  async loadRingAsset(url) {
    const gltf = await new GLTFLoader().loadAsync(url);
    const rawScene = gltf.scene;

    // Object3D has one local transform slot, and this needs two different
    // ones composed: a constant "normalize this asset's own raw units into
    // canonical inner-radius=1.0 space" transform (below), and the per-
    // frame pose transform (position/rotation/scale set in renderFrame).
    // Wrapping the raw glTF scene in a parent group keeps them from
    // colliding - the child gets the constant normalization, the parent
    // (this.ringRoot) gets the per-frame pose.
    rawScene.position.set(-HOLE_CENTER_X / HOLE_INNER_RADIUS, -HOLE_CENTER_Y / HOLE_INNER_RADIUS, 0);
    rawScene.scale.setScalar(1 / HOLE_INNER_RADIUS);

    this.ringRoot = new THREE.Group();
    this.ringRoot.add(rawScene);
    this.ringRoot.visible = false; // hidden until the first valid pose arrives

    // Soft contact shadow: a real ring casts a soft dark shadow onto the
    // skin it rests on - without one, the ring reads as floating just above
    // the finger rather than sitting on it, a common tell for "pasted on"
    // AR overlays. No 3D finger model exists to cast a real shadow onto, so
    // this fakes it: a soft radial-gradient blob, a child of ringRoot (so it
    // inherits the same per-frame pose automatically), offset slightly to
    // the depth-far side and drawn with a lower renderOrder so the ring's
    // own near-side geometry always draws on top of it.
    this._shadowMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({
        map: createSoftShadowTexture(),
        transparent: true,
        depthWrite: false,
        opacity: 0.55,
      }),
    );
    this._shadowMesh.scale.setScalar(2.2);
    this._shadowMesh.position.set(0, -0.35, 0);
    this._shadowMesh.renderOrder = -1;
    this.ringRoot.add(this._shadowMesh);

    this.ringRoot.traverse((obj) => {
      if (!obj.isMesh || obj === this._shadowMesh) return;
      const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
      const materialName = materials[0]?.name?.toLowerCase() ?? "";
      const triCount = (obj.geometry.index?.count ?? obj.geometry.attributes.position.count) / 3;
      // The gem primitive (diamondring.glb's "DIAMONDD" material, ~2.7k
      // tris vs the band's ~1M) is a small lump mounted on ONE side of the
      // band, not a shape symmetric around the finger's circumference like
      // the band is - unlike the band, most of its own local-Y extent is
      // an embedded prong/mount structure that's never meant to be seen,
      // with only a small crown/table cluster meant to face outward. Cut
      // by the same rotating clip plane as the band (same as ring.py's
      // position-based cull - correct, keep that part), it's the DoubleSide
      // backface fallback below that's the actual problem for this piece
      // specifically: on the mount's own un-presentable interior surfaces,
      // showing the backface instead of just letting the clip plane hide
      // them outright reads as broken/wrong geometry (confirmed live: a
      // white blob in one orientation, a dark jagged shard in another).
      // FrontSide alone means those hidden-side fragments simply don't
      // draw - normal backface culling handles it, no reveal needed.
      const isGem = materialName.includes("diamond") || triCount < 20000;
      for (const m of materials) {
        m.clippingPlanes = [this._worldClipPlane];
        m.clipShadows = true;
        m.side = isGem ? THREE.FrontSide : THREE.DoubleSide;
      }
    });
    this.scene.add(this.ringRoot);
  }

  resize(width, height) {
    if (this._width === width && this._height === height) return;
    this._width = width;
    this._height = height;
    this.renderer.setSize(width, height, false);
    // top=0, bottom=height: world Y=0 renders at the screen top, world
    // Y=height at the screen bottom - matches image/pixel Y-down convention
    // directly, no separate flip needed anywhere else.
    this.camera = new THREE.OrthographicCamera(0, width, 0, height, CAMERA_NEAR, CAMERA_FAR);
    this.camera.position.set(0, 0, CAMERA_Z);
    this.camera.lookAt(0, 0, 0);
  }

  /** avgLuminance: 0-255 average gray value sampled from the live video
   * near the ring (see main.js). A ring rendered at a fixed brightness
   * regardless of the room it's actually in is a strong "pasted on" tell -
   * real product photography exposes for the scene it's shot in. Smoothed
   * so exposure doesn't visibly flicker as the sampled region jitters
   * frame to frame. */
  updateExposureFromLuminance(avgLuminance) {
    const target = THREE.MathUtils.clamp(avgLuminance / EXPOSURE_REFERENCE_LUMINANCE, EXPOSURE_MIN, EXPOSURE_MAX);
    this._smoothedExposure += (target - this._smoothedExposure) * EXPOSURE_SMOOTHING;
    this.renderer.toneMappingExposure = this._smoothedExposure;
  }

  /** pose: { centerPx: [x,y], tangent3, normal3, binormal3 (each [x,y,z],
   * from ringPose.js's cameraSpaceFrame), scalePx (target inner radius in
   * pixels) }. Positions/orients/scales the loaded ring mesh and updates
   * the clipping plane to match, then renders one frame. */
  renderFrame(pose) {
    if (!this.camera || !this.ringRoot) return;

    if (pose === null) {
      this.ringRoot.visible = false;
      this.renderer.render(this.scene, this.camera);
      return;
    }

    this.ringRoot.visible = true;
    const { centerPx, tangent3, normal3, binormal3, scalePx } = pose;

    // R's columns = [normal3, binormal3, tangent3]: local X -> normal3
    // (across finger, visible), local Y -> binormal3 (depth), local Z ->
    // tangent3 (along finger) - exactly ring.py's R = np.stack([normal3,
    // binormal3, tangent3], axis=1).
    const basis = new THREE.Matrix4().makeBasis(
      new THREE.Vector3(...normal3),
      new THREE.Vector3(...binormal3),
      new THREE.Vector3(...tangent3),
    );
    this.ringRoot.quaternion.setFromRotationMatrix(basis);
    this.ringRoot.position.set(centerPx[0], centerPx[1], 0);
    this.ringRoot.scale.setScalar(scalePx);
    this.ringRoot.updateMatrixWorld(true);

    this.renderer.render(this.scene, this.camera);
  }
}
