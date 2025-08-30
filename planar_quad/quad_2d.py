
import numpy as np

class Quad2D:
	def __init__(self, m=1.0, I_yy=0.1, g=9.81, r=0.5, dt=0.01):
		self.m = m
		self.I_yy = I_yy
		self.g = g
		self.r = r
		self.dt = dt
		self.reset()

	def reset(self, x=0.0, z=-10.0, theta=0.0, x_dot=0.0, z_dot=0.0, theta_dot=0.0):
		self.x = x
		self.z = z
		self.theta = theta
		self.x_dot = x_dot
		self.z_dot = z_dot
		self.theta_dot = theta_dot
		self.u1 = 0.0
		self.u2 = 0.0
		self.T_z_sp = self.m * self.g
		self.M_y_sp = 0.0
		self.theta_dot_sp = 0.0
		self.theta_sp = 0.0
		self.x_dot_sp = 0.0
		self.z_dot_sp = 0.0
		self.a_x_sp = 0.0
		self.a_z_sp = 0.0
		self.x_sp = 0.0
		self.z_sp = 0.0
		self.mode = 'stab'
		self.K_p_theta_dot = 0.2
		self.K_p_theta = 2.0
		self.K_d = 1.0
		self.k_p_v = 1.0
		self.k_p_x = 0.3
		self.k_p_z = 0.5

	def step(self):
		# Control allocation
		self.u1, self.u2 = np.linalg.solve(
			np.array([[-1, -1], [self.r, -self.r]]),
			np.array([-self.T_z_sp, self.M_y_sp])
		)
		# Physics update
		x_ddot = -(self.u1 + self.u2) * np.sin(self.theta) / self.m
		z_ddot = self.g - (self.u1 + self.u2) * np.cos(self.theta) / self.m
		theta_ddot = self.r * (self.u1 - self.u2) / self.I_yy

		self.x_dot += x_ddot * self.dt
		self.z_dot += z_ddot * self.dt
		self.theta_dot += theta_ddot * self.dt

		self.x += self.x_dot * self.dt
		self.z += self.z_dot * self.dt
		self.theta += self.theta_dot * self.dt

		return self.get_state()

	def get_state(self):
		return np.array([self.x, self.z, self.theta, self.x_dot, self.z_dot, self.theta_dot])

	# --- Control Modes ---
	def crash_control(self):
		return self.T_z_sp, self.M_y_sp

	def rate_control(self):
		self.M_y_sp = self.K_p_theta_dot * (self.theta_dot_sp - self.theta_dot)
		return self.T_z_sp, self.M_y_sp

	def att_control(self):
		self.theta_dot_sp = self.K_p_theta * (self.theta_sp - self.theta)
		return self.rate_control()

	def acceleration_control(self):
		F_x_sp = self.m * self.a_x_sp
		F_z_sp = self.m * (self.g - self.a_z_sp)
		self.T_z_sp = np.sqrt(F_x_sp**2 + F_z_sp**2)
		self.theta_sp = np.arctan2(-F_x_sp, F_z_sp)
		return self.att_control()

	def velocity_control(self):
		error_x = self.x_dot_sp - self.x_dot
		error_z = self.z_dot_sp - self.z_dot
		self.a_x_sp = self.k_p_v * error_x
		self.a_z_sp = self.k_p_v * error_z
		return self.acceleration_control()

	def position_control(self, use_estimated=False, x_est=None, z_est=None):
		if use_estimated and x_est is not None and z_est is not None:
			current_x = x_est
			current_z = z_est
		else:
			current_x = self.x
			current_z = self.z
		error_x = self.x_sp - current_x
		error_z = self.z_sp - current_z
		self.x_dot_sp = self.k_p_x * error_x
		self.z_dot_sp = self.k_p_z * error_z
		return self.velocity_control()

	def set_mode(self, mode):
		self.mode = mode

	def update_control(self, use_estimated=False, x_est=None, z_est=None):
		if self.mode == 'position':
			return self.position_control(use_estimated, x_est, z_est)
		elif self.mode == 'velocity':
			return self.velocity_control()
		elif self.mode == 'acceleration':
			return self.acceleration_control()
		elif self.mode == 'stab':
			return self.att_control()
		elif self.mode == 'acro':
			return self.rate_control()
		else:
			return self.crash_control()
