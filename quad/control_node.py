# remember to run in terminal:
# source ros2_custom_msgs/install/local_setup.bash 

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sr_msgs.msg import Actuators, VehicleAttitude, VehicleAttitudeSetpoint

from std_msgs.msg import Header
import numpy as np
import cv2
from controlers import StabController

class VehicleControlNode(Node):
	def __init__(self):
		super().__init__('vehicle_control_node')
		qos_profile_unity = QoSProfile(
			reliability=ReliabilityPolicy.RELIABLE,
			depth=10
		)

		self.stab_controller = StabController()

		self.vehicle_attitude_sp_sub = self.create_subscription(
			VehicleAttitudeSetpoint,
			'/vehicle_attitude_sp',
			self.vehicle_attitude_sp_callback,
			qos_profile_unity
		)	
		self.vehicle_attitude_sub = self.create_subscription(
			VehicleAttitude,
			'/vehicle_attitude',
			self.vehicle_attitude_callback,
			qos_profile_unity
		)

		self.q_sp = np.array([1.0, 0.0, 0.0, 0.0])
		self.thrust_z = 0.0
		# self.w = np.zeros(3)

		self.actuators_pub = self.create_publisher(Actuators, '/actuators', qos_profile_unity)


	def vehicle_attitude_sp_callback(self, msg):
		self.q_sp = np.array([msg.q_d[0], msg.q_d[1], msg.q_d[2], msg.q_d[3]])
		self.thrust_z = msg.thrust_z

	def vehicle_attitude_callback(self, msg):
		self.q = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
		self.w = np.array(msg.w)

		u = self.stab_controller.compute_control(self.q_sp, self.q, self.w, self.thrust_z)

		msg_actuators = Actuators()
		msg_actuators.timestamp = self.get_clock().now().nanoseconds // 1000_000
		msg_actuators.u = u.astype(np.float32)
		self.actuators_pub.publish(msg_actuators)


def main(args=None):
	rclpy.init(args=args)
	node = VehicleControlNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()

