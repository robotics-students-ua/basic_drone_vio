// Canvas view: world grid, ground, trail, drone, rotor thrust, setpoints,
// and (optionally) the camera: stripes, field of view, landmark rays, pose estimate.
// World frame is NED (x right, z down), so screen y follows z directly.

import { STRIPE, stripeAt, IMAGE_WIDTH, project } from './vision.js';

const MAX_PX_PER_M = 40;
const MIN_VIEW_HEIGHT_M = 16; // zoom out on short screens so ~16 m fit vertically
const CAMERA_LAG = 0.08;      // 0..1, fraction of the gap closed per frame
const GROUND_MARGIN = 48;     // px kept below the ground line
const TOP_MARGIN = 60;        // px kept above the drone
const TOP_MARGIN_IMAGE = 150; // ... when the camera image strip covers the top of the view
const THRUST_ARROW_M = 0.8;  // arrow length (m) for a rotor carrying m*g/2
const STRIPE_BAND_PX = 10;   // height of the striped band on the ground surface
const SKY_RAY_M = 100;       // length of a FOV edge ray that never hits the ground

export class Renderer {
	constructor(canvas) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d');
		this.cam = null;
		this.readColors();
		const media = window.matchMedia('(prefers-color-scheme: dark)');
		media.addEventListener('change', () => this.readColors());
	}

	readColors() {
		const css = getComputedStyle(document.documentElement);
		const v = (name) => css.getPropertyValue(name).trim();
		this.colors = {
			sky: v('--sim-sky'),
			grid: v('--sim-grid'),
			gridMajor: v('--sim-grid-major'),
			label: v('--sim-label'),
			ground: v('--sim-ground'),
			groundLine: v('--sim-ground-line'),
			trail: v('--sim-trail'),
			body: v('--sim-body'),
			nose: v('--sim-nose'),
			thrust: v('--sim-thrust'),
			thrustNeg: v('--sim-thrust-neg'),
			setpoint: v('--sim-setpoint'),
			axisX: v('--sim-axis-x'),
			axisZ: v('--sim-axis-z'),
			stripeDark: v('--sim-stripe-dark'),
			stripeLight: v('--sim-stripe-light'),
			fov: v('--sim-fov'),
			ray: v('--sim-ray'),
			ghost: v('--sim-ghost'),
		};
	}

	resize() {
		const dpr = window.devicePixelRatio || 1;
		const rect = this.canvas.getBoundingClientRect();
		const w = Math.round(rect.width * dpr);
		const h = Math.round(rect.height * dpr);
		if (this.canvas.width !== w || this.canvas.height !== h) {
			this.canvas.width = w;
			this.canvas.height = h;
		}
		this.dpr = dpr;
		this.w = rect.width;
		this.h = rect.height;
		this.ppm = Math.min(MAX_PX_PER_M, this.h / MIN_VIEW_HEIGHT_M);
	}

	// Camera keeps the ground at the bottom while the drone fits, then follows it up
	updateCamera(q, topMargin) {
		const groundPinned = -(this.h / 2 - GROUND_MARGIN) / this.ppm;
		const droneAtTop = q.z + (this.h / 2 - topMargin) / this.ppm;
		const target = { x: q.x, z: Math.min(groundPinned, droneAtTop) };
		if (!this.cam) this.cam = { ...target };
		this.cam.x += (target.x - this.cam.x) * CAMERA_LAG;
		this.cam.z += (target.z - this.cam.z) * CAMERA_LAG;
	}

	toScreen(x, z) {
		return [this.w / 2 + (x - this.cam.x) * this.ppm, this.h / 2 + (z - this.cam.z) * this.ppm];
	}

	// Body point (bx along x_B, bz along z_B) -> world, using R(theta) from chapter 1
	bodyToWorld(q, bx, bz) {
		const c = Math.cos(q.theta), s = Math.sin(q.theta);
		return [q.x + c * bx + s * bz, q.z - s * bx + c * bz];
	}

	draw(world, camera = null) {
		this.resize();
		const q = world.quad;
		this.updateCamera(q, camera ? TOP_MARGIN_IMAGE : TOP_MARGIN);
		const ctx = this.ctx;
		ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
		ctx.fillStyle = this.colors.sky;
		ctx.fillRect(0, 0, this.w, this.h);
		this.drawGrid();
		this.drawGround();
		if (camera) this.drawStripes();
		this.drawXLabels();
		this.drawTrail(world.trail);
		if (camera) this.drawFov(q, camera);
		this.drawSetpoint(q);
		this.drawDrone(q);
		if (camera?.estimate) this.drawEstimate(q, camera);
	}

	drawGrid() {
		const ctx = this.ctx;
		const halfW = this.w / 2 / this.ppm, halfH = this.h / 2 / this.ppm;
		const x0 = Math.floor(this.cam.x - halfW), x1 = Math.ceil(this.cam.x + halfW);
		const z0 = Math.floor(this.cam.z - halfH), z1 = Math.min(0, Math.ceil(this.cam.z + halfH));
		ctx.lineWidth = 1;
		ctx.font = '11px system-ui, sans-serif';
		ctx.fillStyle = this.colors.label;
		for (let x = x0; x <= x1; x++) {
			const [sx] = this.toScreen(x, 0);
			ctx.strokeStyle = x % 5 === 0 ? this.colors.gridMajor : this.colors.grid;
			ctx.beginPath();
			ctx.moveTo(Math.round(sx) + 0.5, 0);
			ctx.lineTo(Math.round(sx) + 0.5, this.h);
			ctx.stroke();
		}
		for (let z = z0; z <= z1; z++) {
			const [, sy] = this.toScreen(0, z);
			ctx.strokeStyle = z % 5 === 0 ? this.colors.gridMajor : this.colors.grid;
			ctx.beginPath();
			ctx.moveTo(0, Math.round(sy) + 0.5);
			ctx.lineTo(this.w, Math.round(sy) + 0.5);
			ctx.stroke();
			if (z % 5 === 0 && z < 0) ctx.fillText(`h = ${-z} м`, 6, sy - 4);
		}
	}

	drawXLabels() {
		const ctx = this.ctx;
		const halfW = this.w / 2 / this.ppm;
		ctx.font = '11px system-ui, sans-serif';
		ctx.fillStyle = this.colors.label;
		ctx.textAlign = 'center';
		for (let x = Math.ceil((this.cam.x - halfW) / 5) * 5; x <= this.cam.x + halfW; x += 5) {
			const [sx, sy] = this.toScreen(x, 0);
			if (sx < 24 || sx > this.w - 24) continue;
			ctx.fillText(`x = ${x}`, sx, Math.min(sy + STRIPE_BAND_PX + 14, this.h - 4));
		}
		ctx.textAlign = 'start';
	}

	drawGround() {
		const ctx = this.ctx;
		const [, sy] = this.toScreen(0, 0);
		if (sy > this.h) return;
		ctx.fillStyle = this.colors.ground;
		ctx.fillRect(0, sy, this.w, this.h - sy);
		ctx.strokeStyle = this.colors.groundLine;
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.moveTo(0, sy);
		ctx.lineTo(this.w, sy);
		ctx.stroke();
	}

	drawTrail(trail) {
		if (trail.length < 2) return;
		const ctx = this.ctx;
		ctx.strokeStyle = this.colors.trail;
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		trail.forEach(([x, z], i) => {
			const [sx, sy] = this.toScreen(x, z);
			if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
		});
		ctx.stroke();
	}

	drawSetpoint(q) {
		const ctx = this.ctx;
		ctx.strokeStyle = this.colors.setpoint;
		ctx.fillStyle = this.colors.setpoint;
		ctx.lineWidth = 2;
		if (q.mode === 'position') {
			const [sx, sy] = this.toScreen(q.x_sp, q.z_sp);
			ctx.beginPath();
			ctx.arc(sx, sy, 8, 0, 2 * Math.PI);
			ctx.moveTo(sx - 14, sy); ctx.lineTo(sx + 14, sy);
			ctx.moveTo(sx, sy - 14); ctx.lineTo(sx, sy + 14);
			ctx.stroke();
		} else if (q.mode === 'velocity') {
			this.arrow(q.x, q.z, q.x + q.x_dot_sp, q.z + q.z_dot_sp, this.colors.setpoint, 2);
		} else if (q.mode === 'acceleration') {
			this.arrow(q.x, q.z, q.x + q.a_x_sp, q.z + q.a_z_sp, this.colors.setpoint, 2);
		} else if (q.mode === 'stab') {
			// dashed outline of the commanded attitude
			const ghost = { x: q.x, z: q.z, theta: q.theta_sp };
			const [ax, az] = this.bodyToWorld(ghost, -q.r * 1.3, 0);
			const [bx, bz] = this.bodyToWorld(ghost, q.r * 1.3, 0);
			const [sa, ta] = this.toScreen(ax, az), [sb, tb] = this.toScreen(bx, bz);
			ctx.setLineDash([5, 4]);
			ctx.beginPath(); ctx.moveTo(sa, ta); ctx.lineTo(sb, tb); ctx.stroke();
			ctx.setLineDash([]);
		}
	}

	drawDrone(q) {
		const ctx = this.ctx;
		const c = this.colors;
		const r = q.r;
		const hover = q.m * q.g / 2;

		// body frame axes
		this.arrow(q.x, q.z, ...this.bodyToWorld(q, r + 0.4, 0), c.axisX, 1.5);
		this.arrow(q.x, q.z, ...this.bodyToWorld(q, 0, 0.6), c.axisZ, 1.5);

		// rotor thrust, along -z_B, length proportional to u_i
		for (const [bx, u] of [[r, q.u1], [-r, q.u2]]) {
			const len = THRUST_ARROW_M * u / hover;
			const base = this.bodyToWorld(q, bx, -0.12);
			const tip = this.bodyToWorld(q, bx, -0.12 - len);
			if (Math.abs(len) > 0.02) this.arrow(...base, ...tip, u >= 0 ? c.thrust : c.thrustNeg, 2.5);
		}

		// arm
		const [lx, lz] = this.toScreen(...this.bodyToWorld(q, -r, 0));
		const [rx, rz] = this.toScreen(...this.bodyToWorld(q, r, 0));
		ctx.strokeStyle = c.body;
		ctx.lineWidth = 5;
		ctx.lineCap = 'round';
		ctx.beginPath(); ctx.moveTo(lx, lz); ctx.lineTo(rx, rz); ctx.stroke();

		// rotors (short bars above each motor)
		ctx.lineWidth = 3;
		for (const bx of [r, -r]) {
			const [p1x, p1z] = this.toScreen(...this.bodyToWorld(q, bx - 0.18, -0.12));
			const [p2x, p2z] = this.toScreen(...this.bodyToWorld(q, bx + 0.18, -0.12));
			const [mx, mz] = this.toScreen(...this.bodyToWorld(q, bx, 0));
			const [tx, tz] = this.toScreen(...this.bodyToWorld(q, bx, -0.12));
			ctx.beginPath(); ctx.moveTo(mx, mz); ctx.lineTo(tx, tz); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(p1x, p1z); ctx.lineTo(p2x, p2z); ctx.stroke();
		}
		ctx.lineCap = 'butt';

		// nose marker on the front rotor (u1)
		const [nx, nz] = this.toScreen(...this.bodyToWorld(q, r, 0));
		ctx.fillStyle = c.nose;
		ctx.beginPath(); ctx.arc(nx, nz, 4.5, 0, 2 * Math.PI); ctx.fill();
	}

	drawStripes() {
		const ctx = this.ctx;
		const [, sy] = this.toScreen(0, 0);
		if (sy > this.h) return;
		const halfW = this.w / 2 / this.ppm;
		const k0 = Math.floor((this.cam.x - halfW) / STRIPE), k1 = Math.ceil((this.cam.x + halfW) / STRIPE);
		for (let k = k0; k <= k1; k++) {
			const [sa] = this.toScreen(k * STRIPE, 0), [sb] = this.toScreen((k + 1) * STRIPE, 0);
			ctx.fillStyle = stripeAt((k + 0.5) * STRIPE) ? this.colors.stripeLight : this.colors.stripeDark;
			ctx.fillRect(sa, sy, sb - sa + 0.5, STRIPE_BAND_PX);
		}
	}

	// Field-of-view wedge and rays to the landmarks the camera sees
	drawFov(q, camera) {
		const ctx = this.ctx;
		const edge = (u) => {
			const hit = camera.groundHit(u, q);
			if (hit !== null) return [hit, 0];
			const c = Math.cos(q.theta), s = Math.sin(q.theta);
			const bx = u / camera.f, n = Math.hypot(bx, 1);
			return [q.x + (c * bx + s) / n * SKY_RAY_M, q.z + (-s * bx + c) / n * SKY_RAY_M];
		};
		const [ox, oz] = this.toScreen(q.x, q.z);
		const [ax, az] = this.toScreen(...edge(-IMAGE_WIDTH / 2));
		const [bx, bz] = this.toScreen(...edge(IMAGE_WIDTH / 2));
		ctx.fillStyle = this.colors.fov;
		ctx.beginPath(); ctx.moveTo(ox, oz); ctx.lineTo(ax, az); ctx.lineTo(bx, bz); ctx.closePath(); ctx.fill();

		ctx.strokeStyle = this.colors.ray;
		ctx.fillStyle = this.colors.ray;
		ctx.lineWidth = 1;
		for (const lx of camera.landmarks) {
			const [sx, sy] = this.toScreen(lx, 0);
			ctx.beginPath(); ctx.moveTo(ox, oz); ctx.lineTo(sx, sy); ctx.stroke();
			ctx.beginPath(); ctx.arc(sx, sy, 3, 0, 2 * Math.PI); ctx.fill();
		}
	}

	// Estimated pose as a ghost drone, with the 2-sigma ellipse of (x, z)
	drawEstimate(q, camera) {
		const ctx = this.ctx;
		const [x, z, theta] = camera.estimate;
		const ghost = { x, z, theta };
		ctx.strokeStyle = this.colors.ghost;
		ctx.fillStyle = this.colors.ghost;

		if (camera.cov) {
			const [[a, b], [, d]] = [camera.cov[0].slice(0, 2), camera.cov[1].slice(0, 2)];
			const tr = (a + d) / 2, det = Math.sqrt(((a - d) / 2) ** 2 + b * b);
			const l1 = tr + det, l2 = Math.max(tr - det, 0);
			const angle = Math.atan2(l1 - a, b || 1e-12);
			const [cx, cz] = this.toScreen(x, z);
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.ellipse(cx, cz, Math.max(2 * Math.sqrt(l1) * this.ppm, 1), Math.max(2 * Math.sqrt(l2) * this.ppm, 1), angle, 0, 2 * Math.PI);
			ctx.stroke();
		}

		const [lx, lz] = this.toScreen(...this.bodyToWorld(ghost, -q.r, 0));
		const [rx, rz] = this.toScreen(...this.bodyToWorld(ghost, q.r, 0));
		ctx.lineWidth = 3;
		ctx.setLineDash([6, 4]);
		ctx.beginPath(); ctx.moveTo(lx, lz); ctx.lineTo(rx, rz); ctx.stroke();
		ctx.setLineDash([]);
		ctx.beginPath(); ctx.arc(rx, rz, 3.5, 0, 2 * Math.PI); ctx.fill();
	}

	arrow(x0, z0, x1, z1, color, width) {
		const ctx = this.ctx;
		const [a, b] = this.toScreen(x0, z0);
		const [c, d] = this.toScreen(x1, z1);
		const len = Math.hypot(c - a, d - b);
		if (len < 2) return;
		const ux = (c - a) / len, uy = (d - b) / len;
		const head = Math.min(9, len * 0.4);
		ctx.strokeStyle = color;
		ctx.fillStyle = color;
		ctx.lineWidth = width;
		ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c - ux * head * 0.8, d - uy * head * 0.8); ctx.stroke();
		ctx.beginPath();
		ctx.moveTo(c, d);
		ctx.lineTo(c - ux * head - uy * head * 0.5, d - uy * head + ux * head * 0.5);
		ctx.lineTo(c - ux * head + uy * head * 0.5, d - uy * head - ux * head * 0.5);
		ctx.closePath();
		ctx.fill();
	}
}

// The camera's 1D image: one row of pixels, stripes as the camera sees them,
// measured landmark edges (ticks) and edges re-projected from the estimate (dots).
export class ImageView {
	constructor(canvas) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d');
		this.readColors();
		window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.readColors());
	}

	readColors() {
		const css = getComputedStyle(document.documentElement);
		const v = (name) => css.getPropertyValue(name).trim();
		this.colors = {
			sky: v('--sim-image-sky'),
			dark: v('--sim-stripe-dark'),
			light: v('--sim-stripe-light'),
			measured: v('--sim-ray'),
			predicted: v('--sim-ghost'),
			centre: v('--sim-axis-z'),
		};
	}

	draw(quad, camera) {
		const dpr = window.devicePixelRatio || 1;
		const rect = this.canvas.getBoundingClientRect();
		if (this.canvas.width !== Math.round(rect.width * dpr)) this.canvas.width = Math.round(rect.width * dpr);
		if (this.canvas.height !== Math.round(rect.height * dpr)) this.canvas.height = Math.round(rect.height * dpr);
		const ctx = this.ctx, w = rect.width, h = rect.height;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, w, h);
		const toU = (px) => (px / w - 0.5) * IMAGE_WIDTH;
		const toPx = (u) => (u / IMAGE_WIDTH + 0.5) * w;

		const band = h - 14;
		for (let px = 0; px < w; px += 1) {
			const hit = camera.groundHit(toU(px + 0.5), quad);
			ctx.fillStyle = hit === null ? this.colors.sky : (stripeAt(hit) ? this.colors.light : this.colors.dark);
			ctx.fillRect(px, 0, 1, band);
		}

		ctx.strokeStyle = this.colors.centre;
		ctx.lineWidth = 1;
		ctx.setLineDash([3, 3]);
		ctx.beginPath(); ctx.moveTo(w / 2 + 0.5, 0); ctx.lineTo(w / 2 + 0.5, band); ctx.stroke();
		ctx.setLineDash([]);

		ctx.fillStyle = this.colors.measured;
		for (const u of camera.measured) ctx.fillRect(toPx(u) - 1, band, 2, 14);

		if (camera.estimate) {
			const [x, z, theta] = camera.estimate;
			ctx.fillStyle = this.colors.predicted;
			for (const lx of camera.landmarks) {
				ctx.beginPath();
				ctx.arc(toPx(project(lx, x, z, theta, camera.f)), band + 7, 3, 0, 2 * Math.PI);
				ctx.fill();
			}
		}
	}
}
