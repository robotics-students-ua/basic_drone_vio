
import numpy as np
import matplotlib.pyplot as plt
import cv2

from quad_2d import Quad2D

# --- Estimator class for state estimation ---
class Estimator:
	def __init__(self, dt=0.01):
		self.dt = dt
		self.reset()

	def reset(self, x0=0.0, z0=-10.0, theta0=0.0):
		self.theta_kf = theta0
		self.P_kf = 1.01
		self.Q_kf = 0.01
		self.R_kf = 20.5
		self.x_est = x0
		self.z_est = z0
		self.x_dot_est = 0.0
		self.z_dot_est = 0.0

	def estimate_theta(self, a_x, a_z):
		# use accelerometer data to estimate theta
		theta_imu = -np.arctan2(a_x, -a_z)
		return theta_imu

	def estimate_theta_kalman(self, a_x, a_z, theta_dot):
		# Predict
		self.theta_kf = self.theta_kf + theta_dot * self.dt
		self.P_kf = self.P_kf + self.Q_kf
		# Measurement update
		theta_meas = self.estimate_theta(a_x, a_z)
		K = self.P_kf / (self.P_kf + self.R_kf)
		self.theta_kf = self.theta_kf + K * (theta_meas - self.theta_kf)
		self.P_kf = (1 - K) * self.P_kf
		return self.theta_kf

	def estimate_pos(self, a_x, a_z, theta):
		# rotate local to global accelerations
		R_LW = np.array([[np.cos(theta), -np.sin(theta)],
						[np.sin(theta),  np.cos(theta)]])
		a_global = R_LW @ np.array([a_x, a_z])
		a_x_est, a_z_est = a_global[0], a_global[1]
		# First integration: acceleration -> velocity
		self.x_dot_est += a_x_est * self.dt
		self.z_dot_est += a_z_est * self.dt
		# Second integration: velocity -> position
		self.x_est += self.x_dot_est * self.dt
		self.z_est += self.z_dot_est * self.dt
		return self.x_est, self.z_est

	def accelerometer(self, a_x_gt, a_z_gt, theta_gt, g=9.81, noise=0.1):
		# rotation matrix from world to local
		R_WL = np.array([[np.cos(theta_gt), np.sin(theta_gt)],
						[-np.sin(theta_gt), np.cos(theta_gt)]])
		a_local = R_WL @ np.array([a_x_gt, a_z_gt - g])
		a_x = a_local[0] + np.random.normal(0, noise)
		a_z = a_local[1] + np.random.normal(0, noise)
		return a_x, a_z

	def gyroscope(self, theta_dot_gt, noise=0.01):
		theta_dot = theta_dot_gt + np.random.normal(0, noise)
		return theta_dot

class Simulator:
	def __init__(self, mode='finite', T_sim=10.0, dt=0.01, plot_states=True):
		self.mode = mode
		self.T_sim = T_sim
		self.dt = dt
		self.plot_states = plot_states
		self.quad = Quad2D(dt=dt)
		self.estimator = Estimator(dt=dt)
		self.states = []
		self.states_est = []
		self.time_steps = []

	def run(self):
		if self.mode == 'finite':
			self.run_finite()
		elif self.mode == 'infinite':
			self.run_infinite()
		else:
			raise ValueError(f"Unknown mode: {self.mode}")

	def run_finite(self):
		num_steps = int(self.T_sim / self.dt)
		self.states = np.zeros((num_steps, 6))
		self.states_est = np.zeros((num_steps, 6))
		self.time_steps = np.arange(0, self.T_sim, self.dt)
		self.x_dot_sp_history = np.zeros(num_steps)
		self.z_dot_sp_history = np.zeros(num_steps)
		self.a_x_sp_history = np.zeros(num_steps)
		self.a_z_sp_history = np.zeros(num_steps)
		self.x_ddot_history = np.zeros(num_steps)
		self.z_ddot_history = np.zeros(num_steps)
		self.theta_sp_history = np.zeros(num_steps)
		self.w_sp_history = np.zeros(num_steps)
		self.u_history = np.zeros((num_steps, 2))
		self.quad.reset()
		self.estimator.reset()
		self.quad.set_mode('stab')
		self.quad.x_sp = 0.0
		self.quad.z_sp = -10.0
		for i in range(num_steps):
			if i == num_steps // 2:
				self.quad.x_sp = 5.0
				self.quad.z_sp = -30.0
				self.quad.set_mode('position')
			self.quad.update_control()
			# Save setpoints and control for plotting
			self.x_dot_sp_history[i] = self.quad.x_dot_sp
			self.z_dot_sp_history[i] = self.quad.z_dot_sp
			self.a_x_sp_history[i] = self.quad.a_x_sp
			self.a_z_sp_history[i] = self.quad.a_z_sp
			self.theta_sp_history[i] = self.quad.theta_sp
			self.w_sp_history[i] = self.quad.theta_dot_sp
			self.u_history[i] = [self.quad.u1, self.quad.u2]
			# Compute accelerations for plotting
			x_ddot = -(self.quad.u1 + self.quad.u2) * np.sin(self.quad.theta) / self.quad.m
			z_ddot = self.quad.g - (self.quad.u1 + self.quad.u2) * np.cos(self.quad.theta) / self.quad.m
			self.x_ddot_history[i] = x_ddot
			self.z_ddot_history[i] = z_ddot
			# --- Estimation step ---
			a_x_imu, a_z_imu = self.estimator.accelerometer(x_ddot, z_ddot, self.quad.theta, g=self.quad.g, noise=0.1)
			theta_dot_imu = self.estimator.gyroscope(self.quad.theta_dot, noise=0.01)
			theta_est = self.estimator.estimate_theta(a_x_imu, a_z_imu)
			theta_est = self.estimator.estimate_theta_kalman(a_x_imu, a_z_imu, theta_dot_imu)
			x_est, z_est = self.estimator.estimate_pos(a_x_imu, a_z_imu, theta_est)
			# Save GT and estimated states
			state = self.quad.step()
			self.states[i] = state
			self.states_est[i] = [x_est, z_est, theta_est, self.estimator.x_dot_est, self.estimator.z_dot_est, self.quad.theta_dot]
		if self.plot_states:
			self.plot_results()

	def run_infinite(self):
		window_width = 1200
		window_height = 600
		scale = 5
		center_x = window_width // 2
		center_y = int(window_height * 0.9)
		self.quad.reset()
		trajectory = []
		# Control variables for interactive mode
		thrust_offset = 0.0
		torque_input = 0.0
		thrust_step = 0.1
		torque_step = 0.001
		a_step = 1
		v_step = 1
		x_step = 1
		z_step = 1
		w_sp_step = 0.01
		theta_step = 0.1
		# Set initial setpoints
		self.quad.x_sp = 0.0
		self.quad.z_sp = -10.0
		self.quad.x_dot_sp = 0.0
		self.quad.z_dot_sp = 0.0
		self.quad.a_x_sp = 0.0
		self.quad.a_z_sp = 0.0
		self.quad.theta_sp = 0.0
		self.quad.theta_dot_sp = 0.0
		mode_list = ['crash', 'acro', 'stab', 'acceleration', 'velocity', 'position']
		mode_index = 2
		self.quad.set_mode(mode_list[mode_index])
		# Boundaries
		z_min = -100
		z_max = 0.0
		x_min = -100
		x_max = 100
		cv2.namedWindow('Quad2D Infinite Sim', cv2.WINDOW_AUTOSIZE)
		while True:
			# Set control based on mode
			mode = self.quad.mode
			if mode == 'crash':
				self.quad.T_z_sp = self.quad.m * self.quad.g + thrust_offset
				self.quad.M_y_sp = torque_input
			elif mode == 'acro':
				self.quad.theta_dot_sp = self.quad.theta_dot_sp
			elif mode == 'stab':
				self.quad.theta_sp = self.quad.theta_sp
			elif mode == 'acceleration':
				self.quad.a_x_sp = self.quad.a_x_sp
				self.quad.a_z_sp = self.quad.a_z_sp
			elif mode == 'velocity':
				self.quad.x_dot_sp = self.quad.x_dot_sp
				self.quad.z_dot_sp = self.quad.z_dot_sp
			elif mode == 'position':
				self.quad.x_sp = self.quad.x_sp
				self.quad.z_sp = self.quad.z_sp
			self.quad.update_control()
			state = self.quad.step()
			# --- Add boundary effects with elastic collision ---
			if self.quad.x <= x_min:
				self.quad.x = x_min
				self.quad.x_dot = -self.quad.x_dot * 0.8
			elif self.quad.x >= x_max:
				self.quad.x = x_max
				self.quad.x_dot = -self.quad.x_dot * 0.8
			if self.quad.z <= z_min:
				self.quad.z = z_min
				self.quad.z_dot = -self.quad.z_dot * 0.8
			elif self.quad.z >= z_max:
				self.quad.z = z_max
				self.quad.z_dot = -self.quad.z_dot * 0.8
			trajectory.append((self.quad.x, self.quad.z))
			# Draw
			img = np.full((window_height, window_width, 3), (64, 64, 64), dtype=np.uint8)
			# Draw simulation bounds
			x_min_screen = int(center_x + x_min * scale)
			x_max_screen = int(center_x + x_max * scale)
			cv2.line(img, (x_min_screen, 0), (x_min_screen, window_height), (255, 0, 0), 2)
			cv2.line(img, (x_max_screen, 0), (x_max_screen, window_height), (255, 0, 0), 2)
			z_min_screen = int(center_y + z_min * scale)
			z_max_screen = int(center_y + z_max * scale)
			cv2.line(img, (0, z_min_screen), (window_width, z_min_screen), (0, 255, 0), 1)
			cv2.line(img, (0, z_max_screen), (window_width, z_max_screen), (0, 255, 0), 1)
			# Draw trajectory
			for i in range(1, len(trajectory)):
				prev = (int(center_x + trajectory[i-1][0]*scale), int(center_y + trajectory[i-1][1]*scale))
				curr = (int(center_x + trajectory[i][0]*scale), int(center_y + trajectory[i][1]*scale))
				cv2.line(img, prev, curr, (0,255,255), 1)
			# Draw drone
			drone_width = int(20 * self.quad.r * scale)
			half_width = drone_width // 2
			cos_theta = np.cos(self.quad.theta)
			sin_theta = -np.sin(self.quad.theta)
			screen_x = int(center_x + self.quad.x * scale)
			screen_y = int(center_y + self.quad.z * scale)
			x1 = int(screen_x - half_width * cos_theta)
			y1 = int(screen_y - half_width * sin_theta)
			x2 = int(screen_x + half_width * cos_theta)
			y2 = int(screen_y + half_width * sin_theta)
			cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 3)
			# Draw thrust arrow
			arrow_dir_x = sin_theta
			arrow_dir_y = -cos_theta
			arrow_length = 20
			arrow_end_x = int(screen_x + arrow_length * arrow_dir_x)
			arrow_end_y = int(screen_y + arrow_length * arrow_dir_y)
			cv2.arrowedLine(img, (screen_x, screen_y), (arrow_end_x, arrow_end_y), (0,255,0), 3, tipLength=0.3)
			# Info
			info_text = [
				f"x: {self.quad.x:.2f}",
				f"z: {self.quad.z:.2f}",
				f"theta: {self.quad.theta:.2f}",
				f"u1: {self.quad.u1:.2f}",
				f"u2: {self.quad.u2:.2f}",
				f"mode: {self.quad.mode}"
			]
			for i, text in enumerate(info_text):
				cv2.putText(img, text, (10, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
			# Setpoints info (upper right)
			setpoints_text = [
				"Current Setpoints:",
				f"Position: x_sp={self.quad.x_sp:.2f}, z_sp={self.quad.z_sp:.2f}",
				f"Velocity: x_dot_sp={self.quad.x_dot_sp:.2f}, z_dot_sp={self.quad.z_dot_sp:.2f}",
				f"Acceleration: a_x_sp={self.quad.a_x_sp:.2f}, a_z_sp={self.quad.a_z_sp:.2f}",
				f"Attitude: theta_sp={self.quad.theta_sp:.3f} rad",
				f"Angular vel: theta_dot_sp={self.quad.theta_dot_sp:.3f} rad/s",
				f"Thrust: T_z_sp={self.quad.T_z_sp:.2f}",
				f"Torque: M_y_sp={self.quad.M_y_sp:.3f}"
			]
			for i, text in enumerate(setpoints_text):
				text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
				x_pos = window_width - text_size[0] - 10
				y_pos = 30 + i * 20
				cv2.putText(img, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
			cv2.imshow('Quad2D Infinite Sim', img)
			key = cv2.waitKey(1) & 0xFF
			if key == 27:  # ESC
				break
			elif key == ord('m'):
				mode_index = (mode_index + 1) % len(mode_list)
				self.quad.set_mode(mode_list[mode_index])
			elif key == 82 or key == ord('w'):  # UP arrow or W key
				if mode in ['crash', 'acro', 'stab']:
					self.quad.T_z_sp += thrust_step
				elif mode == 'acceleration':
					self.quad.a_z_sp -= a_step
				elif mode == 'velocity':
					self.quad.z_dot_sp -= v_step
				elif mode == 'position':
					self.quad.z_sp -= z_step
			elif key == 84 or key == ord('s'):  # DOWN arrow or S key
				if mode in ['crash', 'acro', 'stab']:
					self.quad.T_z_sp -= thrust_step
				elif mode == 'acceleration':
					self.quad.a_z_sp += a_step
				elif mode == 'velocity':
					self.quad.z_dot_sp += v_step
				elif mode == 'position':
					self.quad.z_sp += z_step
			elif key == 81 or key == ord('a'):  # LEFT arrow or A key
				if mode == 'crash':
					self.quad.M_y_sp += torque_step
				elif mode == 'acro':
					self.quad.theta_dot_sp += w_sp_step
				elif mode == 'stab':
					self.quad.theta_sp += theta_step
				elif mode == 'acceleration':
					self.quad.a_x_sp -= a_step
				elif mode == 'velocity':
					self.quad.x_dot_sp -= v_step
				elif mode == 'position':
					self.quad.x_sp -= x_step
			elif key == 83 or key == ord('d'):  # RIGHT arrow or D key
				if mode == 'crash':
					self.quad.M_y_sp -= torque_step
				elif mode == 'acro':
					self.quad.theta_dot_sp -= w_sp_step
				elif mode == 'stab':
					self.quad.theta_sp -= theta_step
				elif mode == 'acceleration':
					self.quad.a_x_sp += a_step
				elif mode == 'velocity':
					self.quad.x_dot_sp += v_step
				elif mode == 'position':
					self.quad.x_sp += x_step
			elif key == ord('r') or key == ord('R'):
				self.quad.reset()
				trajectory = []
				# Reset setpoints
				self.quad.x_sp = 0.0
				self.quad.z_sp = -10.0
				self.quad.x_dot_sp = 0.0
				self.quad.z_dot_sp = 0.0
				self.quad.a_x_sp = 0.0
				self.quad.a_z_sp = 0.0
				self.quad.theta_sp = 0.0
				self.quad.theta_dot_sp = 0.0
				mode_index = 2
				self.quad.set_mode(mode_list[mode_index])
		cv2.destroyAllWindows()

	def plot_results(self):
		fig, ax = plt.subplots(5, 2, figsize=(12, 16))
		fig.tight_layout(pad=3.0)

		# x position
		ax[0, 0].plot(self.time_steps, self.states[:, 0], label='x position (GT)', color='blue')
		ax[0, 0].plot(self.time_steps, self.states_est[:, 0], label='x position (EST)', color='cyan')
		ax[0, 0].set_ylabel('x position (m)')
		ax[0, 0].legend()

		# z position
		ax[0, 1].plot(self.time_steps, self.states[:, 1], label='z position (GT)', color='orange')
		ax[0, 1].plot(self.time_steps, self.states_est[:, 1], label='z position (EST)', color='red')
		ax[0, 1].set_ylabel('z position (m)')
		ax[0, 1].legend()

		# x_dot velocity
		ax[1, 0].plot(self.time_steps, self.states[:, 3], label='x_dot velocity (GT)', color='magenta')
		ax[1, 0].plot(self.time_steps, self.states_est[:, 3], label='x_dot velocity (EST)', color='pink')
		ax[1, 0].plot(self.time_steps, self.x_dot_sp_history, label='x_dot setpoint', linestyle='--', color='red')
		ax[1, 0].set_ylabel('x_dot (m/s)')
		ax[1, 0].legend()

		# z_dot velocity
		ax[1, 1].plot(self.time_steps, self.states[:, 4], label='z_dot velocity (GT)', color='brown')
		ax[1, 1].plot(self.time_steps, self.states_est[:, 4], label='z_dot velocity (EST)', color='yellow')
		ax[1, 1].plot(self.time_steps, self.z_dot_sp_history, label='z_dot setpoint', linestyle='--', color='red')
		ax[1, 1].set_ylabel('z_dot (m/s)')
		ax[1, 1].legend()

		# x_ddot
		ax[2, 0].plot(self.time_steps, self.x_ddot_history, label='x_ddot', color='cyan')
		ax[2, 0].plot(self.time_steps, self.a_x_sp_history, label='x_ddot setpoint', linestyle='--', color='red')
		ax[2, 0].set_ylabel('x_ddot (m/s^2)')
		ax[2, 0].legend()

		# z_ddot
		ax[2, 1].plot(self.time_steps, self.z_ddot_history, label='z_ddot', color='lime')
		ax[2, 1].plot(self.time_steps, self.a_z_sp_history, label='z_ddot setpoint', linestyle='--', color='red')
		ax[2, 1].set_ylabel('z_ddot (m/s^2)')
		ax[2, 1].legend()

		# theta angle
		ax[3, 0].plot(self.time_steps, self.states[:, 2], label='theta angle (GT)', color='green')
		ax[3, 0].plot(self.time_steps, self.states_est[:, 2], label='theta angle (EST)', color='orange')
		ax[3, 0].plot(self.time_steps, self.theta_sp_history, label='theta setpoint', linestyle='--', color='red')
		ax[3, 0].set_ylabel('theta angle (rad)')
		ax[3, 0].legend()

		# angular velocity
		ax[3, 1].plot(self.time_steps, self.states[:, 5], label='angular velocity', color='purple')
		ax[3, 1].plot(self.time_steps, self.w_sp_history, label='setpoint', linestyle='--', color='red')
		ax[3, 1].set_ylabel('theta_dot (rad/s)')
		ax[3, 1].legend()

		# control inputs
		ax[4, 0].plot(self.time_steps[1:], self.u_history[1:, 0], label='u1', color='blue')
		ax[4, 0].plot(self.time_steps[1:], self.u_history[1:, 1], label='u2', color='cyan')
		ax[4, 0].set_ylabel('Control inputs')
		ax[4, 0].set_xlabel('Time (s)')
		ax[4, 0].legend()

		ax[4, 1].axis('off')

		plt.show()

# Example usage:
if __name__ == "__main__":
	# To run finite mode with plots:
	sim = Simulator(mode='infinite')
	# sim = Simulator(mode='finite', T_sim=10.0)
	sim.run()
	# To run infinite mode with OpenCV window:
	# sim = Simulator(mode='infinite')
	# sim.run()
