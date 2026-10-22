// Planar quadrotor: dynamics + cascaded controllers.
// Mirror of 2d_quad/quad_2d.py — keep the two in sync.
// Frame: x right, z down (NED), y toward the viewer, theta > 0 = nose up.

export const MODES = ['crash', 'acro', 'stab', 'acceleration', 'velocity', 'position'];

export class Quad2D {
	constructor({ m = 1.0, I_yy = 0.1, g = 9.81, r = 0.5, dt = 0.01 } = {}) {
		this.m = m;
		this.I_yy = I_yy;
		this.g = g;
		this.r = r;
		this.dt = dt;
		this.reset();
	}

	reset({ x = 0.0, z = -10.0, theta = 0.0, x_dot = 0.0, z_dot = 0.0, theta_dot = 0.0 } = {}) {
		this.x = x;
		this.z = z;
		this.theta = theta;
		this.x_dot = x_dot;
		this.z_dot = z_dot;
		this.theta_dot = theta_dot;
		this.u1 = 0.0;
		this.u2 = 0.0;
		this.T_z_sp = this.m * this.g;
		this.M_y_sp = 0.0;
		this.theta_dot_sp = 0.0;
		this.theta_sp = 0.0;
		this.x_dot_sp = 0.0;
		this.z_dot_sp = 0.0;
		this.a_x_sp = 0.0;
		this.a_z_sp = 0.0;
		this.x_sp = 0.0;
		this.z_sp = 0.0;
		this.mode = 'stab';
		this.K_p_theta_dot = 0.2;
		this.K_p_theta = 2.0;
		this.K_d = 1.0;
		this.k_p_v = 1.0;
		this.k_p_x = 0.3;
		this.k_p_z = 0.5;
	}

	step() {
		// Control allocation: [-1 -1; r -r] [u1 u2]^T = [-T, M]^T
		this.u1 = this.T_z_sp / 2 + this.M_y_sp / (2 * this.r);
		this.u2 = this.T_z_sp / 2 - this.M_y_sp / (2 * this.r);
		// Physics update
		const T = this.u1 + this.u2;
		const x_ddot = -T * Math.sin(this.theta) / this.m;
		const z_ddot = this.g - T * Math.cos(this.theta) / this.m;
		const theta_ddot = this.r * (this.u1 - this.u2) / this.I_yy;

		this.x_dot += x_ddot * this.dt;
		this.z_dot += z_ddot * this.dt;
		this.theta_dot += theta_ddot * this.dt;

		this.x += this.x_dot * this.dt;
		this.z += this.z_dot * this.dt;
		this.theta += this.theta_dot * this.dt;

		return this.getState();
	}

	getState() {
		return [this.x, this.z, this.theta, this.x_dot, this.z_dot, this.theta_dot];
	}

	// --- Control Modes ---
	crashControl() {
		return [this.T_z_sp, this.M_y_sp];
	}

	rateControl() {
		this.M_y_sp = this.K_p_theta_dot * (this.theta_dot_sp - this.theta_dot);
		return [this.T_z_sp, this.M_y_sp];
	}

	attControl() {
		this.theta_dot_sp = this.K_p_theta * (this.theta_sp - this.theta);
		return this.rateControl();
	}

	accelerationControl() {
		const F_x_sp = this.m * this.a_x_sp;
		const F_z_sp = this.m * (this.g - this.a_z_sp);
		this.T_z_sp = Math.hypot(F_x_sp, F_z_sp);
		this.theta_sp = Math.atan2(-F_x_sp, F_z_sp);
		return this.attControl();
	}

	velocityControl() {
		const error_x = this.x_dot_sp - this.x_dot;
		const error_z = this.z_dot_sp - this.z_dot;
		this.a_x_sp = this.k_p_v * error_x;
		this.a_z_sp = this.k_p_v * error_z;
		return this.accelerationControl();
	}

	positionControl() {
		const error_x = this.x_sp - this.x;
		const error_z = this.z_sp - this.z;
		this.x_dot_sp = this.k_p_x * error_x;
		this.z_dot_sp = this.k_p_z * error_z;
		return this.velocityControl();
	}

	setMode(mode) {
		this.mode = mode;
	}

	updateControl() {
		switch (this.mode) {
			case 'position': return this.positionControl();
			case 'velocity': return this.velocityControl();
			case 'acceleration': return this.accelerationControl();
			case 'stab': return this.attControl();
			case 'acro': return this.rateControl();
			default: return this.crashControl();
		}
	}
}
