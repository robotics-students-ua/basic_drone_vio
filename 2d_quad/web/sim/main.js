// UI glue: keyboard, touch pad, mode buttons, readouts, animation loop.

import { MODES } from './quad2d.js';
import { World } from './world.js';
import { Renderer, ImageView } from './render.js';
import { Camera } from './vision.js';

const MODE_INFO = {
	crash: {
		label: 'Crash',
		help: '↑/↓ — тяга T, ←/→ — момент M<sub>y</sub>. Без зворотного зв\'язку.',
	},
	acro: {
		label: 'Acro',
		help: '↑/↓ — тяга T, ←/→ — кутова швидкість θ̇<sub>sp</sub>.',
	},
	stab: {
		label: 'Stab',
		help: '↑/↓ — тяга T, ←/→ — кут нахилу θ<sub>sp</sub>.',
	},
	acceleration: {
		label: 'Accel',
		help: '←/→ — a<sub>x,sp</sub>, ↑/↓ — a<sub>z,sp</sub> (вгору = −z).',
	},
	velocity: {
		label: 'Velocity',
		help: '←/→ — ẋ<sub>sp</sub>, ↑/↓ — ż<sub>sp</sub> (вгору = −z).',
	},
	position: {
		label: 'Position',
		help: 'Стрілки переміщують цільову точку на 1 м.',
	},
};

const KEY_DIRS = {
	ArrowLeft: [-1, 0], KeyA: [-1, 0],
	ArrowRight: [1, 0], KeyD: [1, 0],
	ArrowUp: [0, -1], KeyW: [0, -1],
	ArrowDown: [0, 1], KeyS: [0, 1],
};

const world = new World();
const renderer = new Renderer(document.getElementById('view'));
const camera = new Camera();
const imageView = new ImageView(document.getElementById('image'));
let paused = false;
let visionOn = false;

const params = new URLSearchParams(location.search);
if (MODES.includes(params.get('mode'))) world.setMode(params.get('mode'));

// --- mode buttons ---
const modeBar = document.getElementById('modes');
for (const [i, mode] of MODES.entries()) {
	const b = document.createElement('button');
	b.type = 'button';
	b.dataset.mode = mode;
	b.innerHTML = `<span class="key">${i + 1}</span>${MODE_INFO[mode].label}`;
	b.addEventListener('click', () => { world.setMode(mode); b.blur(); });
	modeBar.appendChild(b);
}

// --- keyboard ---
window.addEventListener('keydown', (e) => {
	if (e.ctrlKey || e.metaKey || e.altKey) return;
	const dir = KEY_DIRS[e.code];
	if (dir) {
		world.input(...dir);
	} else if (/^Digit[1-6]$/.test(e.code)) {
		world.setMode(MODES[Number(e.code.slice(5)) - 1]);
	} else if (e.code === 'KeyM') {
		world.cycleMode();
	} else if (e.code === 'KeyR') {
		resetAll();
	} else if (e.code === 'Space') {
		world.neutral();
	} else if (e.code === 'KeyP') {
		paused = !paused;
	} else {
		return;
	}
	e.preventDefault();
});

// --- touch / mouse pad: repeat while held ---
for (const btn of document.querySelectorAll('[data-dir]')) {
	const [dx, dz] = btn.dataset.dir.split(',').map(Number);
	let timer = null;
	const stop = () => { clearInterval(timer); timer = null; };
	btn.addEventListener('pointerdown', (e) => {
		e.preventDefault();
		world.input(dx, dz);
		stop();
		timer = setInterval(() => world.input(dx, dz), 120);
	});
	for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) btn.addEventListener(ev, stop);
}
document.getElementById('btn-reset').addEventListener('click', (e) => { resetAll(); e.currentTarget.blur(); });
document.getElementById('btn-neutral').addEventListener('click', (e) => { world.neutral(); e.currentTarget.blur(); });
document.getElementById('btn-pause').addEventListener('click', (e) => { paused = !paused; e.currentTarget.blur(); });

function resetAll() {
	world.reset();
	camera.reset();
}

// --- camera controls ---
const $ = (id) => document.getElementById(id);

function setVision(on) {
	visionOn = on;
	$('vision-on').checked = on;
	$('vision-panel').hidden = !on;
	$('image-strip').hidden = !on;
	camera.reset();
}
$('vision-on').addEventListener('change', (e) => { setVision(e.target.checked); e.target.blur(); });

function bindSlider(id, key, format) {
	const input = $(id), out = $(id + '-out');
	input.value = camera[key];
	out.textContent = format(camera[key]);
	input.addEventListener('input', () => {
		camera[key] = Number(input.value);
		out.textContent = format(camera[key]);
	});
	input.addEventListener('change', () => input.blur());
}
bindSlider('fov', 'fovDeg', (v) => `${v}°`);
bindSlider('noise', 'noisePx', (v) => v.toFixed(1));
setVision(params.get('vision') === 'on');

// --- readouts ---
const fmt = (v, d = 2) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d);
const deg = (rad) => fmt(rad * 180 / Math.PI, 1) + '°';

function setpointText(q) {
	switch (q.mode) {
		case 'crash': return `T = ${fmt(q.T_z_sp)} Н, M<sub>y</sub> = ${fmt(q.M_y_sp, 3)} Н·м`;
		case 'acro': return `T = ${fmt(q.T_z_sp)} Н, θ̇<sub>sp</sub> = ${fmt(q.theta_dot_sp)} рад/с`;
		case 'stab': return `T = ${fmt(q.T_z_sp)} Н, θ<sub>sp</sub> = ${deg(q.theta_sp)}`;
		case 'acceleration': return `a<sub>sp</sub> = (${fmt(q.a_x_sp)}, ${fmt(q.a_z_sp)}) м/с²`;
		case 'velocity': return `v<sub>sp</sub> = (${fmt(q.x_dot_sp)}, ${fmt(q.z_dot_sp)}) м/с`;
		case 'position': return `p<sub>sp</sub> = (${fmt(q.x_sp, 1)}, ${fmt(q.z_sp, 1)}) м`;
	}
}

let lastMode = null;
function updatePanel() {
	const q = world.quad;
	if (q.mode !== lastMode) {
		lastMode = q.mode;
		for (const b of modeBar.children) b.classList.toggle('active', b.dataset.mode === q.mode);
		$('help').innerHTML = MODE_INFO[q.mode].help;
	}
	$('setpoint').innerHTML = setpointText(q);
	$('r-t').textContent = fmt(world.t, 1) + ' с';
	$('r-pos').textContent = `${fmt(q.x)}, ${fmt(q.z)} м`;
	$('r-h').textContent = fmt(-q.z) + ' м';
	$('r-vel').textContent = `${fmt(q.x_dot)}, ${fmt(q.z_dot)} м/с`;
	$('r-theta').textContent = deg(q.theta);
	$('r-thetadot').textContent = fmt(q.theta_dot) + ' рад/с';
	$('r-u').textContent = `${fmt(q.u1)}, ${fmt(q.u2)} Н`;
	$('r-u').classList.toggle('warn', q.u1 < 0 || q.u2 < 0);
	const banner = $('banner');
	banner.hidden = !(world.crashed || paused);
	banner.textContent = world.crashed ? 'Аварія! Натисніть R, щоб почати знову' : 'Пауза (P)';
	banner.classList.toggle('crash', world.crashed);
	if (visionOn) updateVisionPanel(q);
}

function updateVisionPanel(q) {
	$('v-n').textContent = camera.landmarks.length;
	const est = camera.estimate;
	if (!est) {
		for (const id of ['v-pos', 'v-theta', 'v-err', 'v-err-theta', 'v-sd', 'v-sd-theta', 'v-corr']) $(id).textContent = '—';
		$('v-n').textContent += camera.landmarks.length < 3 ? ' (потрібно ≥ 3)' : '';
		return;
	}
	$('v-pos').textContent = `${fmt(est[0])}, ${fmt(est[1])} м`;
	$('v-theta').textContent = deg(est[2]);
	$('v-err').textContent = `${fmt((est[0] - q.x) * 100, 1)}, ${fmt((est[1] - q.z) * 100, 1)} см`;
	$('v-err-theta').textContent = deg(est[2] - q.theta);
	if (camera.cov) {
		const sd = [0, 1, 2].map((i) => Math.sqrt(Math.max(camera.cov[i][i], 0)));
		$('v-sd').textContent = `${fmt(sd[0] * 100, 1)}, ${fmt(sd[1] * 100, 1)} см`;
		$('v-sd-theta').textContent = deg(sd[2]);
		$('v-corr').textContent = sd[0] > 0 && sd[2] > 0 ? fmt(camera.cov[0][2] / (sd[0] * sd[2]), 3) : '—';
	}
}

// --- loop ---
let prev = performance.now();
function frame(now) {
	const elapsed = (now - prev) / 1000;
	prev = now;
	if (!paused) {
		world.advance(elapsed);
		if (visionOn) camera.update(world.quad, world.t);
	}
	renderer.draw(world, visionOn ? camera : null);
	if (visionOn) imageView.draw(world.quad, camera);
	updatePanel();
	requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
