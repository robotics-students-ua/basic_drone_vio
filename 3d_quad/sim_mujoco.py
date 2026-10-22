# Quad simulation with MuJoCo rendering (replaces the ROS2 sim_node/control_node + Unity setup).
# Dynamics: our own Quad (NED, RK4). MuJoCo only renders the scene and the onboard camera.
#
# run: python sim_mujoco.py [--viewer] [--camera down|front]
# keys (in the "Camera Feed" window):
#   w/s pitch, a/d roll, q/e yaw, y/h up/down (altitude setpoint or raw thrust),
#   t toggle altitude hold, r reset, esc quit

import argparse
import os
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from quad import Quad
from stab_controller import StabController
from altitude_controller import AltitudeController
from chessboard_pose import ChessboardPoseEstimator

SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scene.xml')

CHESSBOARD_SIZE = (9, 6)  # inner corners
SQUARE_SIZE = 0.25        # m
IMG_W, IMG_H = 640, 480
CAMERA_FPS = 30.0


def ned_to_mj_pos(p):
	# NED (x north, y east, z down) -> MuJoCo world (x forward, y left, z up)
	return np.array([p[0], -p[1], -p[2]])


def ned_to_mj_quat(q):
	# scipy [x, y, z, w] NED/FRD -> MuJoCo [w, x, y, z] z-up/FLU (conjugation by 180 deg about x)
	x, y, z, w = q
	return np.array([w, x, -y, -z])


def build_model():
	spec = mujoco.MjSpec.from_file(SCENE_XML)
	nx, ny = CHESSBOARD_SIZE[0] + 1, CHESSBOARD_SIZE[1] + 1  # squares
	half = SQUARE_SIZE / 2
	# white border, needed for corner detection
	spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
							size=[(nx + 2) * half, (ny + 2) * half, 0.001],
							pos=[0, 0, 0.001], rgba=[1, 1, 1, 1], contype=0, conaffinity=0)
	for i in range(nx):
		for j in range(ny):
			if (i + j) % 2:
				continue
			spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
									size=[half, half, 0.001],
									pos=[(i - (nx - 1) / 2) * SQUARE_SIZE, (j - (ny - 1) / 2) * SQUARE_SIZE, 0.002],
									rgba=[0, 0, 0, 1], contype=0, conaffinity=0)
	return spec.compile()


class QuadSim:
	def __init__(self, viewer=False, camera='down'):
		self.camera = f'{camera}_cam'
		self.dt = 0.004
		self.vehicle = Quad()
		self.vehicle.set_dt(self.dt)
		self.stab_controller = StabController()
		# output normalized by the real max thrust, so thrust_z maps 1:1 to motor commands
		self.altitude_controller = AltitudeController(thrust_accel_max=4 * self.vehicle.T_max / self.vehicle.M)
		self.chessboard_pose_estimator = ChessboardPoseEstimator(CHESSBOARD_SIZE, SQUARE_SIZE, fov_deg=60)

		self.model = build_model()
		self.data = mujoco.MjData(self.model)
		self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W)
		self.viewer = None
		if viewer:
			self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

		self.fps = 0.0
		self.last_time = None
		self.reset()

	def reset(self):
		X = np.zeros(13)
		X[self.vehicle.quaternion_idx] = [0, 0, 0, 1]
		X[self.vehicle.pos_idx] = [0, 0, -3.0]
		self.vehicle.set_initial_state(X)
		self.vehicle.init_solver(self.vehicle.dynamics, type='rk4')
		self.altitude_controller.vz_i = 0.0
		self.alt_hold = True
		self.z_sp = -3.0
		self.thrust_z = self.vehicle.hover_thrust
		self.roll_sp = 0.0
		self.pitch_sp = 0.0
		self.yaw_sp = 0.0

	def control_step(self):
		q = self.vehicle.q
		if self.alt_hold:
			self.thrust_z = self.altitude_controller.update(self.vehicle.p[2], self.z_sp, self.vehicle.v[2],
															q=[q[3], q[0], q[1], q[2]], dt=self.dt)
		q_sp = R.from_euler('xyz', [self.roll_sp, self.pitch_sp, self.yaw_sp]).as_quat()
		u = self.stab_controller.compute_control(
			q_sp=[q_sp[3], q_sp[0], q_sp[1], q_sp[2]],  # [w, x, y, z]
			q=[q[3], q[0], q[1], q[2]],
			w=self.vehicle.w,
			thrust_z=self.thrust_z
		)
		self.vehicle.update_controls(u)
		self.vehicle.dynamics_step()

	def sync_mujoco(self):
		self.data.mocap_pos[0] = ned_to_mj_pos(self.vehicle.p)
		self.data.mocap_quat[0] = ned_to_mj_quat(self.vehicle.q)
		mujoco.mj_forward(self.model, self.data)
		if self.viewer is not None:
			self.viewer.sync()

	def render_camera(self):
		self.renderer.update_scene(self.data, camera=self.camera)
		return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

	def process_frame(self, frame):
		now = time.time()
		if self.last_time is not None:
			dt = now - self.last_time
			instant_fps = 1.0 / dt if dt > 0 else 0.0
			alpha = 0.1  # Smoothing factor (0 < alpha <= 1)
			self.fps = alpha * instant_fps + (1 - alpha) * self.fps
		self.last_time = now

		T_vec, R_mat = self.chessboard_pose_estimator.estimate_pose(frame)
		if T_vec is not None and R_mat is not None:
			self.chessboard_pose_estimator.draw_axes(frame, T_vec, R_mat, axis_length=3.0)

		euler = R.from_quat(self.vehicle.q).as_euler('xyz', degrees=True)
		mode = f"ALT HOLD z_sp={self.z_sp:.2f}" if self.alt_hold else "MANUAL"
		lines = [
			(f"{mode}  Thrust z: {self.thrust_z:.2f}  z: {self.vehicle.p[2]:.2f}", (0, 255, 0)),
			(f"Euler angles [deg]: Roll={euler[0]:.1f}, Pitch={euler[1]:.1f}, Yaw={euler[2]:.1f}", (0, 255, 0)),
			(f"Euler SP [deg]: Roll={np.rad2deg(self.roll_sp):.1f}, Pitch={np.rad2deg(self.pitch_sp):.1f}, Yaw={np.rad2deg(self.yaw_sp):.1f}", (0, 255, 0)),
			(f"FPS: {self.fps:.1f}  t: {self.vehicle.t:.1f}s", (0, 255, 255)),
		]
		if T_vec is not None:
			lines.append((f"Chessboard t [m]: {T_vec[0]:.2f}, {T_vec[1]:.2f}, {T_vec[2]:.2f}", (255, 255, 0)))
		for k, (text, color) in enumerate(lines):
			cv2.putText(frame, text, (10, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
		return frame

	def handle_key(self, key):
		step_angle = np.deg2rad(5)  # rad
		step_thrust = 0.05
		step_alt = 0.25
		if key == 27:
			return False
		elif key == ord('w'):
			self.pitch_sp += step_angle
		elif key == ord('s'):
			self.pitch_sp -= step_angle
		elif key == ord('a'):
			self.roll_sp -= step_angle
		elif key == ord('d'):
			self.roll_sp += step_angle
		elif key == ord('q'):
			self.yaw_sp -= step_angle
		elif key == ord('e'):
			self.yaw_sp += step_angle
		elif key == ord('y'):
			if self.alt_hold:
				self.z_sp -= step_alt
			else:
				self.thrust_z -= step_thrust
		elif key == ord('h'):
			if self.alt_hold:
				self.z_sp += step_alt
			else:
				self.thrust_z += step_thrust
		elif key == ord('t'):
			self.alt_hold = not self.alt_hold
			self.z_sp = self.vehicle.p[2]
			self.altitude_controller.vz_i = 0.0
		elif key == ord('r'):
			self.reset()
		return True

	def run(self):
		steps_per_frame = max(1, round(1.0 / (CAMERA_FPS * self.dt)))
		running = True
		while running:
			frame_start = time.time()
			for _ in range(steps_per_frame):
				self.control_step()
			self.sync_mujoco()
			frame = self.process_frame(self.render_camera())
			cv2.imshow("Camera Feed", frame)
			running = self.handle_key(cv2.waitKey(1) & 0xFF)
			if self.viewer is not None and not self.viewer.is_running():
				running = False
			# keep sim time ~ real time
			sleep = steps_per_frame * self.dt - (time.time() - frame_start)
			if sleep > 0:
				time.sleep(sleep)
		cv2.destroyAllWindows()
		if self.viewer is not None:
			self.viewer.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--viewer', action='store_true', help='also open the interactive MuJoCo viewer')
	parser.add_argument('--camera', choices=['down', 'front'], default='down', help='onboard camera to show')
	args = parser.parse_args()
	QuadSim(viewer=args.viewer, camera=args.camera).run()
