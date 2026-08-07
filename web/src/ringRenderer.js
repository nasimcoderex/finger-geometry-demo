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
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const CAMERA_Z = 10000; // arbitrary large distance - only near/far range matters
const CAMERA_NEAR = 1;
const CAMERA_FAR = 20000;

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

export class RingRenderer {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    this.renderer.localClippingEnabled = true; // required for material.clippingPlanes to take effect
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
    dirLight.position.set(0, 0, 1); // camera-collocated, matches ring.py's shading convention
    this.scene.add(dirLight);

    // The band/gem materials are near-zero roughness, high metalness (the
    // Python side inspected the glTF material: roughness=0.05, metallic
    // unset -> glTF default 1.0). A metalness~1 PBR material renders almost
    // black under direct lighting alone - it needs something to reflect.
    // RoomEnvironment is Three.js's cheap built-in stand-in for a real
    // HDRI, generated once via PMREMGenerator - no external asset needed.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment()).texture;
    pmrem.dispose();

    this.camera = null; // built in resize(), once we know the video's pixel dimensions
    this.ringRoot = null; // Object3D wrapping the loaded GLTF scene
    this._worldClipPlane = new THREE.Plane();
    // local Y=0 plane: THREE.Plane(normal, constant) keeps points where
    // normal.dot(p) + constant >= 0, i.e. keeps local Y >= 0 - matching
    // ring.py's `all_avg_depth > 0` cull test exactly (local Y maps to
    // world Z / depth via the camera-space frame's binormal3 axis, so this
    // hides the far half of the ring, the part that should be behind the
    // finger, while keeping the near/visible half).
    this._localClipPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

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

    this.ringRoot.traverse((obj) => {
      if (!obj.isMesh) return;
      const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const m of materials) {
        m.clippingPlanes = [this._worldClipPlane];
        m.clipShadows = true;
        m.side = THREE.DoubleSide; // avoid seeing through to nothing at the clip boundary
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

    // Three.js does not auto-attach clipping planes to a moving object's
    // local space - the plane's normal/constant must be re-derived in
    // world space every frame from the object's current transform.
    this._worldClipPlane.copy(this._localClipPlane).applyMatrix4(this.ringRoot.matrixWorld);

    this.renderer.render(this.scene, this.camera);
  }
}
