// A cheap, self-contained stand-in for a real HDRI, tuned for jewelry
// rather than Three.js's generic RoomEnvironment - same technique (a Scene
// fed into PMREMGenerator.fromScene), just lit like an actual product-photo
// softbox rig instead of a random room: a large bright key light above-
// front (the classic 45-degree jewelry softbox position), a cooler weaker
// fill from the opposite side (keeps shadows from going pure black without
// flattening the metal's form), and a small warm rim light behind to catch
// facet edges - real product photography for rings almost always uses
// exactly this 3-light pattern because it's what makes facets sparkle
// instead of reading as a flat gray blob.
import { BackSide, BoxGeometry, Mesh, MeshLambertMaterial, MeshStandardMaterial, Scene } from "three";

function areaLightMaterial(colorHex, intensity) {
  return new MeshLambertMaterial({ color: 0x000000, emissive: colorHex, emissiveIntensity: intensity });
}

export class JewelryEnvironment extends Scene {
  constructor() {
    super();

    const geometry = new BoxGeometry();
    geometry.deleteAttribute("uv");

    // enclosing room - neutral light gray so bounced light stays neutral,
    // not tinted; keeps metal looking like metal, not "gray room" colored
    const room = new Mesh(geometry, new MeshStandardMaterial({ side: BackSide, color: 0xc8c8c8 }));
    room.scale.set(30, 30, 30);
    this.add(room);

    // key light: large, bright, above and slightly in front - the
    // dominant light, casts the main specular streak across the band/gem
    const key = new Mesh(geometry, areaLightMaterial(0xfff4e0, 60));
    key.position.set(6, 10, 8);
    key.scale.set(8, 0.2, 6);
    key.rotation.x = -0.5;
    this.add(key);

    // fill: cooler, much weaker, opposite side - stops the far side of the
    // ring from going flat black, without competing with the key light
    const fill = new Mesh(geometry, areaLightMaterial(0xdce8ff, 14));
    fill.position.set(-8, 2, -6);
    fill.scale.set(6, 6, 0.2);
    this.add(fill);

    // rim: small, warm, behind - catches facet edges/silhouette, the
    // detail that makes a cut gem read as "sparkly" rather than "shiny gray"
    const rim = new Mesh(geometry, areaLightMaterial(0xffe0b0, 35));
    rim.position.set(-2, 6, -12);
    rim.scale.set(3, 3, 0.2);
    this.add(rim);
  }

  dispose() {
    const resources = new Set();
    this.traverse((obj) => {
      if (obj.isMesh) {
        resources.add(obj.geometry);
        resources.add(obj.material);
      }
    });
    for (const r of resources) r.dispose();
  }
}
