import * as THREE from 'three';
import { STOPS } from '../campaign/chapter1';
import { pauses, pointAt, WAYPOINTS } from './path';
import type { Look, Rail } from './path';

/** The campaign path in 3D: a night mountain under the Spirit Moon, lantern-lit, foggy, with embers. */
export class World3D {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(70, 1, 0.1, 600);
  private embers: THREE.Points;
  private scrolls = new Map<number, THREE.Group>();
  private arenas = new Map<number, THREE.Mesh>();
  private lanterns: THREE.PointLight[] = [];
  private moonMesh: THREE.Mesh;
  private t = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.6;
    // a lighter blue-violet haze that fades the distant mountains rather than hiding the view
    this.scene.background = new THREE.Color('#332f5c');
    this.scene.fog = new THREE.FogExp2('#3a3768', 0.0048);
    // strong moonlight + a bright hemisphere fill so the night reads clearly, not pitch dark
    this.scene.add(new THREE.HemisphereLight('#aab0ff', '#3a2e3a', 1.1));
    const moon = new THREE.DirectionalLight('#c9d2ff', 2.0);
    moon.position.set(-80, 120, -60);
    this.scene.add(moon);
    this.scene.add(new THREE.AmbientLight('#6a6698', 0.35));
    this.moonMesh = this.buildSky();
    this.buildTerrain();
    this.buildPath();
    this.buildPlaces();
    this.embers = this.buildEmbers();
    this.resize();
  }

  resize(): void {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setTaken(stops: number[]): void {
    for (const [i, g] of this.scrolls) g.visible = !stops.includes(i);
  }

  render(rail: Rail, look: Look, dt: number, highlight: 'scroll' | 'arena' | null): void {
    this.t += dt;
    const { pos, heading } = rail.pose();
    const bob = Math.sin(this.t * 7) * 0.04;
    this.camera.position.set(pos.x, pos.y + 1.7 + bob, pos.z);
    this.camera.rotation.set(look.pitch, heading + look.yaw, 0, 'YXZ');
    // embers drift up and wrap around the camera
    const p = this.embers.geometry.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < p.count; i++) {
      let y = p.getY(i) + dt * (0.4 + (i % 7) * 0.08);
      if (y > pos.y + 12) y = pos.y - 2;
      p.setY(i, y);
    }
    p.needsUpdate = true;
    this.embers.position.set(pos.x, 0, pos.z);
    // scrolls turn and pulse; arena rings flicker, brighter when you're at one
    for (const g of this.scrolls.values()) { g.rotation.y += dt * 0.8; g.position.y = g.userData.baseY + Math.sin(this.t * 2) * 0.1; }
    for (const m of this.arenas.values()) (m.material as THREE.MeshBasicMaterial).opacity = (highlight === 'arena' ? 0.9 : 0.55) + 0.15 * Math.sin(this.t * 9);
    // lantern flicker
    for (let i = 0; i < this.lanterns.length; i++) {
      const l = this.lanterns[i];
      l.intensity = 9 + Math.sin(this.t * 9 + i * 2.1) * 0.8 + Math.sin(this.t * 23 + i) * 0.4;
    }
    // the moon breathes a slow glow
    (this.moonMesh.material as THREE.MeshBasicMaterial).opacity = 0.85 + 0.1 * Math.sin(this.t * 0.5);
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void { this.renderer.dispose(); }

  /** A big inverted sphere with a vertical gradient (canvas texture): a moonlit-night sky, not a flat void. */
  private buildSkyDome(): void {
    const c = document.createElement('canvas');
    c.width = 2;
    c.height = 256;
    const ctx = c.getContext('2d')!;
    const g = ctx.createLinearGradient(0, 0, 0, 256);
    g.addColorStop(0, '#171335'); // zenith: deep night violet
    g.addColorStop(0.55, '#332f5c'); // mid sky
    g.addColorStop(0.82, '#5c548f'); // low sky, lit by the moon
    g.addColorStop(1, '#7a729c'); // horizon glow
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 2, 256);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const dome = new THREE.Mesh(
      new THREE.SphereGeometry(500, 24, 16),
      new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, fog: false, depthWrite: false }),
    );
    this.scene.add(dome);
  }

  private buildSky(): THREE.Mesh {
    this.buildSkyDome();
    // large and clearly visible from the path: placed toward -z (the way the path mostly faces),
    // off to one side, low enough to sit over the valley rather than straight overhead
    const moon = new THREE.Mesh(new THREE.SphereGeometry(34, 32, 16), new THREE.MeshBasicMaterial({ color: '#eef1ff', fog: false, transparent: true, opacity: 0.95 }));
    moon.position.set(-120, 95, -360);
    this.scene.add(moon);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(58, 32, 16), new THREE.MeshBasicMaterial({ color: '#9aa0ff', transparent: true, opacity: 0.22, fog: false }));
    halo.position.copy(moon.position);
    this.scene.add(halo);
    const stars = new THREE.BufferGeometry(), n = 900, a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const th = Math.random() * Math.PI * 2, ph = Math.random() * 0.45 * Math.PI, r = 450;
      a.set([Math.cos(th) * Math.cos(ph) * r, Math.sin(ph) * r + 40, Math.sin(th) * Math.cos(ph) * r], i * 3);
    }
    stars.setAttribute('position', new THREE.BufferAttribute(a, 3));
    this.scene.add(new THREE.Points(stars, new THREE.PointsMaterial({ color: '#ffffff', size: 1.2, fog: false })));
    return moon;
  }

  /**
   * The ground: a ridge/cliff side rather than a corridor. One side (downhill, `dx < 0`) drops away
   * into an open valley so there's a clear vista down the mountain; the other rises gently, well
   * back from the path, toward the distant peaks — it never walls in the path close beside it.
   */
  private buildTerrain(): void {
    const geo = new THREE.PlaneGeometry(900, 900, 160, 160);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      const along = Math.max(0, Math.min(300, -z));
      const onPath = pointAt(along);
      const dx = x - onPath.x;
      const noise = Math.sin(x * 0.05) * 3 + Math.cos(z * 0.04) * 4;
      let h: number;
      if (dx < 0) {
        // cliff side: falls away into the valley, opening up the view
        const drop = Math.max(0, -dx - 5);
        h = onPath.y - 1.5 - drop * 1.1 - Math.min(drop * 0.25, 55) + noise;
      } else {
        // uphill side: stays open near the path, then rises slowly toward the far peaks
        const rise = Math.max(0, dx - 22);
        h = onPath.y - 1.5 + rise * 0.22 + noise;
      }
      pos.setY(i, h);
    }
    geo.computeVertexNormals();
    this.scene.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: '#3a3452', roughness: 0.95, flatShading: true })));
    for (let i = 0; i < 9; i++) {
      const peak = new THREE.Mesh(new THREE.ConeGeometry(70 + i * 10, 170 + (i % 3) * 50, 5), new THREE.MeshStandardMaterial({ color: '#2c2740', flatShading: true }));
      peak.position.set(-320 + i * 80, 10, -480 - (i % 2) * 70);
      this.scene.add(peak);
    }
  }

  /** The stone path: flagstones along the waypoints, a faint glowing edge, lanterns every few metres. */
  private buildPath(): void {
    const stone = new THREE.MeshStandardMaterial({ color: '#8b84a0', roughness: 0.85 });
    const edgeMat = new THREE.MeshBasicMaterial({ color: '#ffb35c', transparent: true, opacity: 0.35 });
    for (let d = 0; d < 300; d += 1.2) {
      const a = pointAt(d), b = pointAt(d + 1.2);
      const slab = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.25, 1.1), stone);
      slab.position.set(a.x, a.y, a.z);
      slab.lookAt(b.x, a.y, b.z);
      this.scene.add(slab);
      // a thin glowing seam along each edge of the stone
      for (const side of [-1, 1]) {
        const seam = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.02, 1.05), edgeMat);
        seam.position.set(a.x, a.y + 0.14, a.z);
        seam.lookAt(b.x, a.y + 0.14, b.z);
        seam.translateX(side * 1.62);
        this.scene.add(seam);
      }
    }
    // lantern count kept modest to stay under the point-light budget
    for (let d = 8; d < 300; d += 20) {
      const a = pointAt(d), side = (Math.floor(d / 20) % 2) * 2 - 1;
      this.addLantern(a.x + side * 2.4, a.y, a.z);
    }
  }

  private addLantern(x: number, y: number, z: number): void {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 1.6), new THREE.MeshStandardMaterial({ color: '#3a2618' }));
    post.position.set(x, y + 0.8, z);
    const glow = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.45, 0.35), new THREE.MeshBasicMaterial({ color: '#ffb35c' }));
    glow.position.set(x, y + 1.75, z);
    const light = new THREE.PointLight('#ff9a4a', 9, 16, 2);
    light.position.copy(glow.position);
    this.lanterns.push(light);
    this.scene.add(post, glow, light);
  }

  /** Each stop's place, its scroll stand (if any) and its arena ring of fire. */
  private buildPlaces(): void {
    const at = (d: number) => pointAt(d);
    // temple courtyard: red pillars, a brazier, dummies
    const c = at(STOPS[0].pathAt);
    const red = new THREE.MeshStandardMaterial({ color: '#8c1f1a', roughness: 0.6 });
    for (const [dx, dz] of [[-6, 4], [6, 4], [-6, -8], [6, -8]]) {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.5, 6), red);
      p.position.set(c.x + dx, c.y + 3, c.z + dz);
      this.scene.add(p);
    }
    const roof = new THREE.Mesh(new THREE.ConeGeometry(11, 3, 4), new THREE.MeshStandardMaterial({ color: '#2a1a1a' }));
    roof.position.set(c.x, c.y + 7.5, c.z - 2);
    roof.rotation.y = Math.PI / 4;
    this.scene.add(roof);
    this.addBrazier(c.x, c.y, c.z - 6);
    // bamboo along the bridge
    const bamboo = new THREE.MeshStandardMaterial({ color: '#4d6b3a' });
    for (let d = 95; d < 140; d += 1.6) {
      for (const side of [-1, 1]) {
        const a = at(d), stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 5 + (d % 3)), bamboo);
        stalk.position.set(a.x + side * (3 + (d % 2)), a.y + 2.5, a.z);
        this.scene.add(stalk);
      }
    }
    // stone garden: boulders on raked sand
    const g = at(STOPS[3].pathAt);
    const sand = new THREE.Mesh(new THREE.CircleGeometry(12, 32), new THREE.MeshStandardMaterial({ color: '#6d6456' }));
    sand.rotation.x = -Math.PI / 2;
    sand.position.set(g.x + 8, g.y + 0.05, g.z);
    this.scene.add(sand);
    for (let i = 0; i < 6; i++) {
      const b = new THREE.Mesh(new THREE.DodecahedronGeometry(0.8 + (i % 3) * 0.5), new THREE.MeshStandardMaterial({ color: '#4c4852', flatShading: true }));
      b.position.set(g.x + 4 + (i * 2.3) % 9, g.y + 0.5, g.z - 5 + (i * 3.1) % 10);
      this.scene.add(b);
    }
    // the village gate: palisade, torches, banners
    const gate = at(STOPS[4].pathAt + 8);
    const wood = new THREE.MeshStandardMaterial({ color: '#4a3220' });
    for (let i = -8; i <= 8; i++) {
      if (Math.abs(i) < 2) continue;
      const log = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 4.5), wood);
      log.position.set(gate.x + i * 0.55, gate.y + 2.2, gate.z);
      this.scene.add(log);
    }
    this.addBrazier(gate.x - 2.5, gate.y, gate.z + 1);
    this.addBrazier(gate.x + 2.5, gate.y, gate.z + 1);
    // scroll stands and arenas
    STOPS.forEach((s, i) => {
      if (s.scrollAt !== undefined) this.addScroll(i, at(s.scrollAt));
      this.addArena(i, at(s.pathAt));
    });
  }

  private addBrazier(x: number, y: number, z: number): void {
    const bowl = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.35, 0.6, 12), new THREE.MeshStandardMaterial({ color: '#3b3036', metalness: 0.4 }));
    bowl.position.set(x, y + 1, z);
    const fire = new THREE.Mesh(new THREE.ConeGeometry(0.45, 1, 8), new THREE.MeshBasicMaterial({ color: '#ff8a2a' }));
    fire.position.set(x, y + 1.8, z);
    const light = new THREE.PointLight('#ff7a30', 12, 20, 2);
    light.position.set(x, y + 2, z);
    this.lanterns.push(light);
    this.scene.add(bowl, fire, light);
  }

  private addScroll(stop: number, p: { x: number; y: number; z: number }): void {
    const g = new THREE.Group();
    const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.4, 1), new THREE.MeshStandardMaterial({ color: '#3d2a1c' }));
    stand.position.y = -0.6;
    const roll = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.7, 12), new THREE.MeshBasicMaterial({ color: '#ffd27a' }));
    roll.rotation.z = Math.PI / 2;
    const light = new THREE.PointLight('#ffcc66', 8, 10, 2);
    g.add(stand, roll, light);
    g.position.set(p.x + 1.8, p.y + 1.2, p.z);
    g.userData.baseY = g.position.y;
    this.scene.add(g);
    this.scrolls.set(stop, g);
  }

  private addArena(stop: number, p: { x: number; y: number; z: number }): void {
    const ring = new THREE.Mesh(new THREE.RingGeometry(2.6, 3, 48), new THREE.MeshBasicMaterial({ color: '#ff6a2a', transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(p.x, p.y + 0.15, p.z - 3);
    this.scene.add(ring);
    this.arenas.set(stop, ring);
  }

  private buildEmbers(): THREE.Points {
    const n = 400, a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) a.set([(Math.random() - 0.5) * 40, Math.random() * 60, (Math.random() - 0.5) * 40], i * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(a, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: '#ffa04a', size: 0.12, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.scene.add(pts);
    return pts;
  }
}

/** For the map: the path's outline from above, as 2D points in 0..1. */
export function pathOutline(): { x: number; y: number }[] {
  const xs = WAYPOINTS.map(p => p.x), zs = WAYPOINTS.map(p => p.z);
  const [x0, x1, z0, z1] = [Math.min(...xs) - 10, Math.max(...xs) + 10, Math.min(...zs), Math.max(...zs)];
  return Array.from({ length: 61 }, (_, i) => { const p = pointAt((i / 60) * 300); return { x: (p.x - x0) / (x1 - x0), y: (p.z - z1) / (z0 - z1) }; });
}
export { pauses };
