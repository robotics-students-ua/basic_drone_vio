// Downward camera + pose from landmarks (chapter 4). Mirrors vision_1.py.
// Camera frame = body frame: optical axis z_B, image line along x_B.
// Landmarks are the edges of ground stripes at x = k * STRIPE, z = 0.

export const STRIPE = 1.2;          // m, stripe width (3.dynamics_online_vision.py)
export const IMAGE_WIDTH = 2 * 400 * Math.tan(Math.PI / 6); // px, f = 400 at 60° FOV
const GN_ITERS = 5;
const MIN_LANDMARKS = 3;

// Ground colour under x: 0 = dark stripe, 1 = light stripe
export const stripeAt = (x) => ((Math.floor(x / STRIPE) % 2) + 2) % 2;

function gaussian() {
	let u = 0, v = 0;
	while (u === 0) u = Math.random();
	while (v === 0) v = Math.random();
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Landmark l = [lx, 0] -> camera frame: R(theta)^T (l - p)
function toCamera(lx, x, z, theta) {
	const c = Math.cos(theta), s = Math.sin(theta);
	const dx = lx - x, dz = -z;
	return [c * dx - s * dz, s * dx + c * dz];
}

export function project(lx, x, z, theta, f) {
	const [xc, zc] = toCamera(lx, x, z, theta);
	return f * xc / zc;
}

// Row of d u / d [x, z, theta]: k * [-h/rho², Δ/rho², -1], k = f (1 + (u/f)²)
function jacobianRow(lx, x, z, theta, f) {
	const h = -z;
	const delta = lx - x;
	const rho2 = delta * delta + h * h;
	const u = project(lx, x, z, theta, f);
	const k = f * (1 + (u / f) ** 2);
	return [-k * h / rho2, k * delta / rho2, -k];
}

function solve3(A, b) {
	const M = A.map((row, i) => [...row, b[i]]);
	for (let col = 0; col < 3; col++) {
		let piv = col;
		for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
		[M[col], M[piv]] = [M[piv], M[col]];
		if (Math.abs(M[col][col]) < 1e-12) return null;
		for (let r = 0; r < 3; r++) {
			if (r === col) continue;
			const k = M[r][col] / M[col][col];
			for (let c = col; c < 4; c++) M[r][c] -= k * M[col][c];
		}
	}
	return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}

function inverse3(A) {
	const cols = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map((e) => solve3(A, e));
	if (cols.some((c) => !c)) return null;
	return [0, 1, 2].map((i) => [cols[0][i], cols[1][i], cols[2][i]]);
}

export class Camera {
	constructor() {
		this.fovDeg = 60;
		this.noisePx = 1.0;
		this.rate = 30; // Hz
		this.reset();
	}

	get f() {
		return (IMAGE_WIDTH / 2) / Math.tan((this.fovDeg * Math.PI / 180) / 2);
	}

	reset() {
		this.lastT = -Infinity;
		this.landmarks = [];    // visible landmark x positions
		this.measured = [];     // noisy u_i, px
		this.estimate = null;   // [x, z, theta]
		this.cov = null;        // 3x3
		this.lost = true;
	}

	// Call after each physics advance; captures at `rate` Hz of sim time
	update(quad, t) {
		if (t - this.lastT < 1 / this.rate - 1e-9) return false;
		this.lastT = t;
		this.capture(quad);
		this.solve(quad);
		return true;
	}

	visibleLandmarks(x, z, theta) {
		if (z >= 0) return [];
		const f = this.f;
		// stripe edges inside the FOV footprint on the ground (wide search, then filter)
		const h = -z;
		const reach = h * 8 + 5;
		const k0 = Math.ceil((x - reach) / STRIPE), k1 = Math.floor((x + reach) / STRIPE);
		const out = [];
		for (let k = k0; k <= k1; k++) {
			const lx = k * STRIPE;
			const [xc, zc] = toCamera(lx, x, z, theta);
			if (zc <= 1e-6) continue;
			if (Math.abs(f * xc / zc) <= IMAGE_WIDTH / 2) out.push(lx);
		}
		return out;
	}

	capture(quad) {
		const f = this.f;
		this.landmarks = this.visibleLandmarks(quad.x, quad.z, quad.theta);
		this.measured = this.landmarks.map((lx) => project(lx, quad.x, quad.z, quad.theta, f) + this.noisePx * gaussian());
	}

	// Gauss-Newton from the previous estimate (or the truth when lost),
	// with known data association (we know which edge is which)
	solve(quad) {
		const n = this.landmarks.length;
		if (n < MIN_LANDMARKS) {
			this.lost = true;
			this.estimate = null;
			this.cov = null;
			return;
		}
		const f = this.f;
		let s = this.lost || !this.estimate ? [quad.x, quad.z, quad.theta] : [...this.estimate];
		let JtJ = null;
		for (let it = 0; it < GN_ITERS; it++) {
			JtJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
			const Jtr = [0, 0, 0];
			for (let i = 0; i < n; i++) {
				const lx = this.landmarks[i];
				const r = this.measured[i] - project(lx, s[0], s[1], s[2], f);
				const J = jacobianRow(lx, s[0], s[1], s[2], f);
				for (let a = 0; a < 3; a++) {
					Jtr[a] += J[a] * r;
					for (let b = 0; b < 3; b++) JtJ[a][b] += J[a] * J[b];
				}
			}
			const step = solve3(JtJ, Jtr);
			if (!step) { this.lost = true; this.estimate = null; this.cov = null; return; }
			s = s.map((v, i) => v + step[i]);
			if (Math.hypot(...step) < 1e-9) break;
		}
		const inv = inverse3(JtJ);
		const sigma2 = Math.max(this.noisePx, 1e-3) ** 2;
		this.cov = inv ? inv.map((row) => row.map((v) => v * sigma2)) : null;
		this.estimate = s;
		this.lost = s[1] >= 0;
	}

	// For drawing the image: ground x hit by the ray through pixel u (null = sky)
	groundHit(u, quad) {
		const c = Math.cos(quad.theta), s = Math.sin(quad.theta);
		const bx = u / this.f, bz = 1;               // ray in camera frame
		const wx = c * bx + s * bz, wz = -s * bx + c * bz; // R(theta) * ray
		if (wz <= 1e-6 || quad.z >= 0) return null;
		const t = -quad.z / wz;
		return quad.x + wx * t;
	}
}
