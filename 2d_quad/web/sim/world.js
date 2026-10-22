// Simulation wrapper around Quad2D: fixed-step clock, ground contact,
// pilot input per mode, trail. Kept free of DOM code.

import { Quad2D, MODES } from './quad2d.js';

// Step sizes per key press, as in 2.dynamics_online.py
export const STEPS = {
	thrust: 0.1,   // N
	torque: 0.001, // N·m
	rate: 0.01,    // rad/s
	angle: 0.1,    // rad
	accel: 1.0,    // m/s²
	vel: 1.0,      // m/s
	pos: 1.0,      // m
};

const CRASH_SPEED = 3.0;   // m/s, vertical touchdown speed that counts as a crash
const CRASH_ANGLE = 0.8;   // rad, |theta| at touchdown that counts as a crash
const GROUND_FRICTION = 0.95;
const TRAIL_LENGTH = 3000;

export class World {
	constructor() {
		this.quad = new Quad2D();
		this.reset();
	}

	reset() {
		this.quad.reset();
		this.quad.T_z_sp = this.quad.m * this.quad.g;
		this.t = 0;
		this.accumulator = 0;
		this.crashed = false;
		this.onGround = false;
		this.trail = [];
		this.setMode(this.quad.mode);
	}

	// Switching mode starts the new mode from a neutral, safe setpoint
	setMode(mode) {
		if (!MODES.includes(mode)) return;
		const q = this.quad;
		q.setMode(mode);
		switch (mode) {
			case 'crash': q.M_y_sp = 0; break;
			case 'acro': q.theta_dot_sp = 0; break;
			case 'stab': q.theta_sp = 0; break;
			case 'acceleration': q.a_x_sp = 0; q.a_z_sp = 0; break;
			case 'velocity': q.x_dot_sp = 0; q.z_dot_sp = 0; break;
			case 'position': q.x_sp = Math.round(q.x); q.z_sp = Math.min(Math.round(q.z), -1); break;
		}
	}

	cycleMode() {
		const i = MODES.indexOf(this.quad.mode);
		this.setMode(MODES[(i + 1) % MODES.length]);
	}

	// Pilot input: dx, dz in {-1, 0, 1}; dx = +1 means "right", dz = -1 means "up"
	input(dx, dz) {
		const q = this.quad;
		switch (q.mode) {
			case 'crash':
				q.T_z_sp -= dz * STEPS.thrust;
				q.M_y_sp -= dx * STEPS.torque; // right = nose down = negative moment
				break;
			case 'acro':
				q.T_z_sp -= dz * STEPS.thrust;
				q.theta_dot_sp -= dx * STEPS.rate;
				break;
			case 'stab':
				q.T_z_sp -= dz * STEPS.thrust;
				q.theta_sp -= dx * STEPS.angle;
				break;
			case 'acceleration':
				q.a_x_sp += dx * STEPS.accel;
				q.a_z_sp += dz * STEPS.accel;
				break;
			case 'velocity':
				q.x_dot_sp += dx * STEPS.vel;
				q.z_dot_sp += dz * STEPS.vel;
				break;
			case 'position':
				q.x_sp += dx * STEPS.pos;
				q.z_sp = Math.min(q.z_sp + dz * STEPS.pos, 0);
				break;
		}
	}

	// "Let go of the sticks": neutral setpoint for the active mode
	neutral() {
		const q = this.quad;
		if (['crash', 'acro', 'stab'].includes(q.mode)) q.T_z_sp = q.m * q.g;
		this.setMode(q.mode);
	}

	// Advance by wall-clock time `elapsed` (s) using fixed steps of quad.dt
	advance(elapsed) {
		if (this.crashed) return;
		const dt = this.quad.dt;
		this.accumulator += Math.min(elapsed, 0.1);
		while (this.accumulator >= dt) {
			this.stepOnce();
			this.accumulator -= dt;
			if (this.crashed) break;
		}
	}

	stepOnce() {
		const q = this.quad;
		q.updateControl();
		q.step();
		this.t += q.dt;
		this.groundContact();
		const last = this.trail[this.trail.length - 1];
		if (!last || Math.hypot(q.x - last[0], q.z - last[1]) > 0.05) {
			this.trail.push([q.x, q.z]);
			if (this.trail.length > TRAIL_LENGTH) this.trail.shift();
		}
	}

	groundContact() {
		const q = this.quad;
		this.onGround = q.z >= 0;
		if (!this.onGround) return;
		if (q.z_dot > CRASH_SPEED || Math.abs(q.theta) > CRASH_ANGLE) {
			this.crashed = true;
		}
		q.z = 0;
		q.z_dot = Math.min(q.z_dot, 0);
		q.x_dot *= GROUND_FRICTION;
		if (!this.crashed) {
			// lands level on its skids
			q.theta = 0;
			q.theta_dot = 0;
		}
	}
}
