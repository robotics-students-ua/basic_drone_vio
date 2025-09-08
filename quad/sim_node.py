# remember to run in terminal:
# source ros2_custom_msgs/install/local_setup.bash 

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sr_msgs.msg import VehicleSim, Actuators, VehicleAttitude, VehicleAttitudeSetpoint
from sensor_msgs.msg import CompressedImage  # Import the necessary message type

from std_msgs.msg import Header
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import time
from quad import Quad
from chessboard_pose import ChessboardPoseEstimator

class VehicleSimNode(Node):
	
	def __init__(self):
		super().__init__('vehicle_sim_node')
		qos_profile_unity = QoSProfile(
			reliability=ReliabilityPolicy.RELIABLE,
			depth=10
		)
		
		self.vehicle = Quad()
		self.dt = 0.004
		self.vehicle.set_dt(self.dt)
		self.vehicle.init_solver(self.vehicle.dynamics, type='rk4')
		self.vehicle.set_position(np.array([0, 0, -100]))

		self.chessboard_pose_estimator = ChessboardPoseEstimator()

		# Create subscribers
		self.camera_image_subscriber = self.create_subscription(
			CompressedImage,
			'/camera_image',
			self.camera_image_callback,
			qos_profile_unity
		)

		self.actuators_subscriber = self.create_subscription(
			Actuators,
			'/actuators',
			self.actuators_callback,
			qos_profile_unity
		)

		self.last_time = None
		self.fps = 0.0
		self._fps_times = []
		self._fps_last_update = time.time()
		self.vehicle_sim_pub = self.create_publisher(VehicleSim, 'vehicle_sim', qos_profile_unity)
		self.vehicle_attitude_pub = self.create_publisher(VehicleAttitude, 'vehicle_attitude', qos_profile_unity)
		self.vehicle_attitude_sp_pub = self.create_publisher(VehicleAttitudeSetpoint, 'vehicle_attitude_sp', qos_profile_unity)

		self.frame = None
		self.u = np.array([0.0, 0.0, 0.0, 0.0])

		self.thrust_z = -self.vehicle.M * 9.81
		self.roll_sp = 0.0
		self.pitch_sp = 0.0	
		self.yaw_sp = 0.0

		self.last_time = None
		self.fps = 0.0

		mode = "unity"
		# mode = "python"
		# if mode == "unity":
		# elif mode == "python":
		self.timer = self.create_timer(self.dt, self.publish_sim_data)  # 4ms
		if mode == "python":
			self.timer_sp = self.create_timer(self.dt, self.timer_setpoint)

		self.sim_start_time = time.time()  # Initialize simulation start time

	def actuators_callback(self, msg):
		self.u = np.array(msg.u)
		self.vehicle.update_controls(self.u)
		# print(self.u)

	def publish_sim_data(self):
		self.vehicle.dynamics_step()
		X = self.vehicle.get_state()
		msg = VehicleSim()
		msg.timestamp = self.get_clock().now().nanoseconds // 1000_000  # microseconds
		msg.x = float(X[0])
		msg.y = float(X[1])
		msg.z = float(X[2])
		# quat is represented as [x, y, z, w]
		# but msgs expects a different order [w, x, y, z]
		msg.q = [float(X[9]), float(X[6]), float(X[7]), float(X[8])]
		self.vehicle_sim_pub.publish(msg)

		msg_att = VehicleAttitude()
		msg_att.timestamp = msg.timestamp
		msg_att.q = [float(X[9]), float(X[6]), float(X[7]), float(X[8])]
		msg_att.w = [float(X[10]), float(X[11]), float(X[12])]
		self.vehicle_attitude_pub.publish(msg_att)

	
	def timer_setpoint(self,):
		elapsed = time.time() - self.sim_start_time
		if 2.0 <= elapsed < 4.0:  # Step from 5s to 6s
			self.pitch_sp = 0.2    # Step value in radians
		else:
			self.pitch_sp = 0.0    # Baseline value
		q_sp = R.from_euler('xyz', [self.roll_sp, self.pitch_sp, self.yaw_sp]).as_quat()
		self.publish_setpoint(q_sp, self.thrust_z)

	def publish_setpoint(self, q_sp, T_z):
		msg = VehicleAttitudeSetpoint()
		msg.timestamp = self.get_clock().now().nanoseconds // 1000_000  # microseconds
		msg.q_d = [float(q_sp[3]), float(q_sp[0]), float(q_sp[1]), float(q_sp[2])]
		msg.thrust_z = float(T_z)
		self.vehicle_attitude_sp_pub.publish(msg)

	def camera_image_callback(self, msg):
		np_arr = np.frombuffer(msg.data, np.uint8)
		frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		self.frame = frame
		if self.frame is None:
			print("No frame received yet.")
			return
		
		now = time.time()
		if hasattr(self, 'last_time') and self.last_time is not None:
			dt = now - self.last_time
			instant_fps = 1.0 / dt if dt > 0 else 0.0
			alpha = 0.1  # Smoothing factor (0 < alpha <= 1)
			self.fps = alpha * instant_fps + (1 - alpha) * getattr(self, 'fps', instant_fps)
		else:
			self.fps = 0.0
		self.last_time = now

		T_vec, R_mat = self.chessboard_pose_estimator.estimate_pose(self.frame)
		if T_vec is not None and R_mat is not None:
			self.chessboard_pose_estimator.draw_axes(self.frame, T_vec, R_mat)
		# print(f"Pose: {T_vec}, {R_mat}")

		step_angle = np.deg2rad(5) # rad
		step_thrust = 0.5


		euler = R.from_quat(self.vehicle.q).as_euler('xyz', degrees=True)

		# Overlay info on frame
		frame_disp = self.frame.copy()
		text1 = f"Thrust z: {self.thrust_z:.2f}"
		text2 = f"Euler angles [deg]: Roll={euler[0]:.1f}, Pitch={euler[1]:.1f}, Yaw={euler[2]:.1f}"
		text3 = f"Euler SP [deg]: Roll={np.rad2deg(self.roll_sp):.1f}, Pitch={np.rad2deg(self.pitch_sp):.1f}, Yaw={np.rad2deg(self.yaw_sp):.1f}"
		text4 = f"FPS: {self.fps:.1f}"
		cv2.putText(frame_disp, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
		cv2.putText(frame_disp, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
		cv2.putText(frame_disp, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
		cv2.putText(frame_disp, text4, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

		cv2.imshow("Camera Feed", frame_disp)
		key = cv2.waitKey(1)
		# esc to exit
		if key == 27:
			cv2.destroyAllWindows()
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
			self.thrust_z -= step_thrust
		elif key == ord('h'):
			self.thrust_z += step_thrust
		
		# print(f"roll_sp: {roll_sp}, pitch_sp: {pitch_sp}, yaw_sp: {yaw_sp}, thrust_z: {self.thrust_z}")
		self.q_sp = R.from_euler('xyz', [self.roll_sp, self.pitch_sp, self.yaw_sp]).as_quat()
		self.publish_setpoint(self.q_sp, self.thrust_z)


def main(args=None):
	rclpy.init(args=args)
	node = VehicleSimNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()

