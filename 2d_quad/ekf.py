import numpy as np
import matplotlib.pyplot as plt

from quad_2d import Quad2D

# Chapter 5: Kalman filter -> EKF.
# Frames as in chapter 1: x right, z down (NED), theta > 0 nose up,
# R(theta) = [[c, s], [-s, c]] maps body -> world.
# The accelerometer measures specific force f_B = R^T (a - g), g = [0, G].

G = 9.81


def rot(theta):
	c, s = np.cos(theta), np.sin(theta)
	return np.array([[c, s], [-s, c]])


def rot_dtheta(theta):
	c, s = np.cos(theta), np.sin(theta)
	return np.array([[-s, c], [-c, -s]])


# --- Sensors (with bias, unlike simulator.py) ---
class IMU:
	def __init__(self, gyro_noise=0.01, gyro_bias=0.02, acc_noise=0.1, acc_bias=(0.2, -0.1), rng=None):
		self.gyro_noise = gyro_noise
		self.gyro_bias = gyro_bias
		self.acc_noise = acc_noise
		self.acc_bias = np.array(acc_bias)
		self.rng = rng or np.random.default_rng()

	def read(self, a_world, theta, theta_dot):
		f_body = rot(theta).T @ (a_world - np.array([0.0, G]))
		acc = f_body + self.acc_bias + self.rng.normal(0, self.acc_noise, 2)
		gyro = theta_dot + self.gyro_bias + self.rng.normal(0, self.gyro_noise)
		return acc, gyro


class GPS:
	def __init__(self, noise=0.5, rate=5.0, rng=None):
		self.noise = noise
		self.rate = rate
		self.rng = rng or np.random.default_rng()

	def read(self, x, z):
		return np.array([x, z]) + self.rng.normal(0, self.noise, 2)


class Barometer:
	def __init__(self, noise=0.5, rate=10.0, rng=None):
		self.noise = noise
		self.rate = rate
		self.rng = rng or np.random.default_rng()

	def read(self, z):
		return z + self.rng.normal(0, self.noise)


# --- Linear Kalman filter: vertical channel, IMU + barometer ---
class AltitudeKF:
	"""
	State [z, z_dot, b_az]. Assumes the drone is level, so the vertical
	accelerometer reading is a_z = f_z + G (+ bias). Barometer measures z.
	"""
	def __init__(self, dt, acc_noise, baro_noise, bias_walk=1e-3, z0=-10.0):
		self.dt = dt
		self.x = np.array([z0, 0.0, 0.0])
		self.P = np.diag([1.0, 0.1, 0.5**2])
		self.F = np.array([[1, dt, -dt**2 / 2],
						   [0, 1, -dt],
						   [0, 0, 1]])
		self.B = np.array([dt**2 / 2, dt, 0])
		g = np.array([dt**2 / 2, dt, 0])
		self.Q = np.outer(g, g) * acc_noise**2 + np.diag([0, 0, bias_walk**2 * dt])
		self.H = np.array([[1.0, 0, 0]])
		self.R = np.array([[baro_noise**2]])

	def predict(self, f_z):
		a_z = f_z + G
		self.x = self.F @ self.x + self.B * a_z
		self.P = self.F @ self.P @ self.F.T + self.Q

	def update(self, z_baro):
		y = np.array([z_baro]) - self.H @ self.x
		S = self.H @ self.P @ self.H.T + self.R
		K = self.P @ self.H.T @ np.linalg.inv(S)
		self.x = self.x + K @ y
		self.P = (np.eye(3) - K @ self.H) @ self.P


# --- EKF: full planar state, IMU prediction + GPS position update ---
class PlanarEKF:
	"""
	State s = [x, z, theta, vx, vz, b_g, b_ax, b_az] (with_bias=True)
	   or s = [x, z, theta, vx, vz]                  (with_bias=False).
	Prediction is driven by the IMU, GPS corrects position.
	"""
	def __init__(self, dt, gyro_noise, acc_noise, gps_noise, with_bias=True,
				 gyro_bias_walk=1e-4, acc_bias_walk=1e-3, s0=None):
		self.dt = dt
		self.with_bias = with_bias
		self.n = 8 if with_bias else 5
		self.s = np.zeros(self.n) if s0 is None else np.array(s0, dtype=float)
		p0 = [1.0, 1.0, np.radians(5)**2, 0.5**2, 0.5**2]
		if with_bias:
			p0 += [0.05**2, 0.5**2, 0.5**2]
		self.P = np.diag(p0)
		self.gyro_noise = gyro_noise
		self.acc_noise = acc_noise
		self.gyro_bias_walk = gyro_bias_walk
		self.acc_bias_walk = acc_bias_walk
		self.H = np.zeros((2, self.n))
		self.H[0, 0] = self.H[1, 1] = 1.0
		self.R = np.eye(2) * gps_noise**2

	def biases(self):
		if self.with_bias:
			return self.s[5], self.s[6:8]
		return 0.0, np.zeros(2)

	def predict(self, acc, gyro):
		dt = self.dt
		b_g, b_a = self.biases()
		theta = self.s[2]
		f = acc - b_a
		a_world = rot(theta) @ f + np.array([0.0, G])

		# Jacobian of the continuous dynamics, A = d s_dot / d s
		A = np.zeros((self.n, self.n))
		A[0, 3] = A[1, 4] = 1.0
		A[3:5, 2] = rot_dtheta(theta) @ f
		if self.with_bias:
			A[2, 5] = -1.0
			A[3:5, 6:8] = -rot(theta)
		F = np.eye(self.n) + A * dt

		# Process noise: gyro noise drives theta, accel noise drives v, biases random-walk
		Q = np.zeros((self.n, self.n))
		Q[2, 2] = self.gyro_noise**2 * dt
		Q[3:5, 3:5] = np.eye(2) * self.acc_noise**2 * dt
		if self.with_bias:
			Q[5, 5] = self.gyro_bias_walk**2 * dt
			Q[6:8, 6:8] = np.eye(2) * self.acc_bias_walk**2 * dt

		# Nonlinear state propagation
		self.s[0:2] += self.s[3:5] * dt
		self.s[2] += (gyro - b_g) * dt
		self.s[3:5] += a_world * dt
		self.P = F @ self.P @ F.T + Q

	def update_gps(self, pos):
		y = pos - self.H @ self.s
		S = self.H @ self.P @ self.H.T + self.R
		K = self.P @ self.H.T @ np.linalg.inv(S)
		self.s = self.s + K @ y
		I_KH = np.eye(self.n) - K @ self.H
		self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form, keeps P symmetric
		return y, S


class DeadReckoning:
	"""Chapter 3: integrate the gyro and the accelerometer, no corrections."""
	def __init__(self, dt, s0):
		self.dt = dt
		self.s = np.array(s0, dtype=float)

	def predict(self, acc, gyro):
		dt = self.dt
		a_world = rot(self.s[2]) @ acc + np.array([0.0, G])
		self.s[0:2] += self.s[3:5] * dt
		self.s[2] += gyro * dt
		self.s[3:5] += a_world * dt


# --- Scenario: fly waypoints with the true-state controller, estimate open loop ---
WAYPOINTS = [(0, -10), (10, -10), (10, -20), (-5, -15), (0, -10), (8, -12)]
SEGMENT = 10.0  # s per waypoint


def run_planar(T_sim=60.0, dt=0.01, seed=0):
	rng = np.random.default_rng(seed)
	quad = Quad2D(dt=dt)
	quad.reset()
	quad.set_mode('position')
	imu = IMU(rng=rng)
	gps = GPS(rng=rng)
	s0 = [quad.x, quad.z, quad.theta, quad.x_dot, quad.z_dot]
	filters = {
		'dead_reckoning': DeadReckoning(dt, s0),
		'ekf_no_bias': PlanarEKF(dt, imu.gyro_noise, imu.acc_noise, gps.noise, with_bias=False, s0=s0),
		'ekf_bias': PlanarEKF(dt, imu.gyro_noise, imu.acc_noise, gps.noise, with_bias=True, s0=s0 + [0, 0, 0]),
	}
	gps_every = int(round(1 / (gps.rate * dt)))
	n = int(T_sim / dt)
	log = {'t': np.arange(n) * dt, 'truth': np.zeros((n, 5)), 'gps': []}
	for k in filters:
		log[k] = np.zeros((n, filters[k].n if hasattr(filters[k], 'n') else 5))
	log['ekf_bias_sd'] = np.zeros((n, 8))
	log['innovations'] = []

	for i in range(n):
		wp = WAYPOINTS[min(int(i * dt / SEGMENT), len(WAYPOINTS) - 1)]
		quad.x_sp, quad.z_sp = wp
		quad.update_control()
		# true acceleration over this step (same model as Quad2D.step)
		T = quad.T_z_sp
		a_world = np.array([-T * np.sin(quad.theta) / quad.m, G - T * np.cos(quad.theta) / quad.m])
		acc, gyro = imu.read(a_world, quad.theta, quad.theta_dot)
		quad.step()
		for f in filters.values():
			f.predict(acc, gyro)
		if i % gps_every == 0:
			z = gps.read(quad.x, quad.z)
			log['gps'].append((i * dt, *z))
			filters['ekf_no_bias'].update_gps(z)
			y, S = filters['ekf_bias'].update_gps(z)
			log['innovations'].append(y @ np.linalg.solve(S, y))
		log['truth'][i] = [quad.x, quad.z, quad.theta, quad.x_dot, quad.z_dot]
		for k, f in filters.items():
			log[k][i] = f.s
		log['ekf_bias_sd'][i] = np.sqrt(np.diag(filters['ekf_bias'].P))
	log['true_bias'] = (imu.gyro_bias, imu.acc_bias)
	return log


def run_altitude(T_sim=60.0, dt=0.01, seed=0):
	rng = np.random.default_rng(seed)
	imu = IMU(rng=rng, acc_bias=(0.0, 0.3))
	baro = Barometer(rng=rng)
	kf = AltitudeKF(dt, imu.acc_noise, baro.noise)
	z, vz = -10.0, 0.0
	baro_every = int(round(1 / (baro.rate * dt)))
	n = int(T_sim / dt)
	log = np.zeros((n, 7))
	for i in range(n):
		t = i * dt
		a_z = -0.5 * np.sin(0.5 * t)  # gentle vertical motion
		acc, _ = imu.read(np.array([0.0, a_z]), 0.0, 0.0)
		z += vz * dt
		vz += a_z * dt
		kf.predict(acc[1])
		if i % baro_every == 0:
			kf.update(baro.read(z))
		log[i] = [t, z, vz, *kf.x, np.sqrt(kf.P[0, 0])]
	return log, imu.acc_bias[1]


def rms(e):
	return np.sqrt(np.mean(np.square(e)))


def report():
	log, bias = run_altitude()
	late = log[:, 0] > 20
	print('--- Altitude KF (IMU + barometer, sigma 0.5 m) ---')
	print(f'z error RMS (t>20 s): {rms(log[late, 3] - log[late, 1]):.3f} m, filter sigma_z: {log[-1, 6]:.3f} m')
	print(f'vz error RMS (t>20 s): {rms(log[late, 4] - log[late, 2]):.3f} m/s')
	print(f'accel bias: true {bias:.3f}, estimated {log[-1, 5]:.3f}')

	log = run_planar()
	late = log['t'] > 20
	truth = log['truth'][late]
	print('\n--- Planar: IMU + GPS (sigma 0.5 m, 5 Hz), RMS for t > 20 s ---')
	for k in ['dead_reckoning', 'ekf_no_bias', 'ekf_bias']:
		est = log[k][late]
		pos = rms(np.linalg.norm(est[:, 0:2] - truth[:, 0:2], axis=1))
		th = np.degrees(rms(est[:, 2] - truth[:, 2]))
		vel = rms(np.linalg.norm(est[:, 3:5] - truth[:, 3:5], axis=1))
		print(f'{k:15s} position {pos:8.2f} m   theta {th:7.2f} deg   velocity {vel:6.2f} m/s')
	gps = np.array(log['gps'])
	gps_late = gps[gps[:, 0] > 20]
	idx = np.round(gps_late[:, 0] / 0.01).astype(int)
	print(f'{"raw GPS":15s} position {rms(np.linalg.norm(gps_late[:, 1:3] - log["truth"][idx, 0:2], axis=1)):8.2f} m')
	b_g, b_a = log['true_bias']
	s = log['ekf_bias'][-1]
	sd = log['ekf_bias_sd'][-1]
	print(f'gyro bias:  true {b_g:.4f}, est {s[5]:.4f} +- {sd[5]:.4f} rad/s')
	print(f'accel bias: true {b_a}, est [{s[6]:.3f}, {s[7]:.3f}] +- [{sd[6]:.3f}, {sd[7]:.3f}] m/s^2')
	nis = np.array(log['innovations'])
	print(f'mean NIS: {nis[len(nis) // 3:].mean():.2f} (expected 2 for a 2D measurement)')
	return log


def plot(log):
	t = log['t']
	fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
	labels = {'dead_reckoning': 'IMU only', 'ekf_no_bias': 'EKF, no bias states', 'ekf_bias': 'EKF with biases'}
	for k, lbl in labels.items():
		ax[0].plot(t, log[k][:, 0] - log['truth'][:, 0], label=lbl)
		ax[1].plot(t, np.degrees(log[k][:, 2] - log['truth'][:, 2]), label=lbl)
	sd = log['ekf_bias_sd']
	ax[0].fill_between(t, -2 * sd[:, 0], 2 * sd[:, 0], color='gray', alpha=0.3, label='EKF +-2 sigma')
	ax[1].fill_between(t, -2 * np.degrees(sd[:, 2]), 2 * np.degrees(sd[:, 2]), color='gray', alpha=0.3)
	ax[0].set_ylim(-5, 5)
	ax[1].set_ylim(-10, 10)
	ax[0].set_ylabel('x error (m)')
	ax[1].set_ylabel('theta error (deg)')
	b_g, b_a = log['true_bias']
	ax[2].plot(t, log['ekf_bias'][:, 5], label='b_g est')
	ax[2].axhline(b_g, color='C0', linestyle='--')
	ax[2].plot(t, log['ekf_bias'][:, 6], label='b_ax est')
	ax[2].axhline(b_a[0], color='C1', linestyle='--')
	ax[2].plot(t, log['ekf_bias'][:, 7], label='b_az est')
	ax[2].axhline(b_a[1], color='C2', linestyle='--')
	ax[2].set_ylabel('bias')
	ax[2].set_xlabel('time (s)')
	for a in ax:
		a.legend(loc='upper right')
		a.grid(True)
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	plot(report())
