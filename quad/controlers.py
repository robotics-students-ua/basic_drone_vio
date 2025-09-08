import numpy as np




class StabController:
    def __init__(self):
        self.K_p = np.array([100.0, 100.0, 30.0])
        self.K_d = np.array([51, 50.5, 70.5])
        self.max_tilt = np.radians(30)  # max tilt angle in radians
        self.T_sp = 0.0
        self.M_sp = np.zeros(3)

    def compute_control(self, q_sp, q, w, thrust_z):
        """
        q_sp: desired quaternion [w, x, y, z]
        q: current quaternion [w, x, y, z]
        thrust_z: desired vertical thrust
        
        """
        # Quaternion error
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        q_err = self.quat_multiply(q_sp, q_conj)
        if q_err[0] < 0:
            q_err = -q_err  # Ensure shortest path

        # Convert quaternion error to axis-angle
        angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
        if angle > np.pi:
            angle -= 2 * np.pi
        axis = q_err[1:]
        if np.linalg.norm(axis) > 1e-8:
            axis /= np.linalg.norm(axis)
        else:
            axis = np.zeros(3)
        euler_pitch = np.arctan2(2 * (q[0] * q[2] + q[1] * q[3]), 1 - 2 * (q[2]**2 + q[3]**2))
        euler_sp_pitch = np.arctan2(2 * (q_sp[0] * q_sp[2] + q_sp[1] * q_sp[3]), 1 - 2 * (q_sp[2]**2 + q_sp[3]**2))
        # print(f'Angle (deg): {np.degrees(angle)}, Axis: {axis}')
        # print(f"euler pitch error {np.rad2deg(euler_pitch - euler_sp_pitch)}")
        # Desired angular velocity from PD control
        w_desired = self.K_p * angle * axis 


        self.T_sp = thrust_z
        # self.M_sp = self.K_d * (w_desired - w)
        self.M_sp = self.K_p * angle * axis + self.K_d * (-w)
        # print(self.M_sp)
        u = self.control_allocation(self.T_sp, self.M_sp)

        return u

    def control_allocation(self, T_sp, M_sp):
        # given T_sp, M_sp return actuators u1 u2 u3 u4 for quad
        effectiveness_matrix = np.array([[ 1,  1,  1,  1],   # total thrust
                                          [ -1, 1, 1,  -1],   # roll (x)
                                          [1, -1,  1,  -1],   # pitch (y)
                                          [ 1, 1,  -1, -1]])  # yaw (z)
        return np.linalg.pinv(effectiveness_matrix) @ np.array([T_sp, *M_sp])

    def quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])