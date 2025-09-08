import numpy as np
from scipy.spatial.transform import Rotation as R
from rigid_body import RigidBody
from sensors import IMU, Barometer, Magnetometer

class Quad(RigidBody):
    def __init__(self, X_init=None, mass=None, J=None) -> None:
        super().__init__()
        self.u = np.zeros(4) 
        # self.T_max = 10  # max thrust from one motor
        self.T_max = 1  # max thrust from one motor
        self.L = 0.1 # length of an arm
        self.Q_max = 0.2 # moment from one motor
        self.aero = None
        if X_init is None:
            X_init = np.zeros(13)
            X_init[self.quaternion_idx] = R.from_euler('xyz', [0, 0, 0]).as_quat()
            # X_init[2] = -10
        if mass is None:
            mass = 1.0
        if J is None:
            J = np.eye(3)

        self.set_initial_state(X_init)
        self.set_mass(mass)
        self.set_inertia(J)
        self.init_solver(self.dynamics, type='rk4')
        self.IMU = IMU()
        self.barometer = Barometer()
        # self.magnetometer = Magnetometer(50.45, 30.52, 0)

    @property
    def motors(self): return self.u
    

    def update_controls(self, controls):
        if controls is not None:
            self.u = controls

    def get_motor_thrust(self):
        # return np.array([0, 0, - self.T_max * (self.u[0] + self.u[1] + self.u[2]  + self.u[3])] ) 
        return np.array([0, 0,  self.T_max * (self.u[0] + self.u[1] + self.u[2]  + self.u[3])] ) 
    
    def get_motor_moment(self):
         return np.array([self.L * self.T_max * (-self.u[0] + self.u[1] + self.u[2]  - self.u[3]),
                           self.L * self.T_max * (self.u[0] - self.u[1] + self.u[2]  - self.u[3]),
                          self.Q_max * (self.u[0] + self.u[1]  - self.u[2]  - self.u[3])] ) 
    
    def dynamics(self, t, X, U=None):
        if self.aero is not None:
            self.F_a, self.M_a = self.aero.get_aero_force_moment(self.v, self.q, self.w, self.servos)
        self.F_m = self.get_motor_thrust()
        self.M_m = self.get_motor_moment()
        # print(f'F_m: {self.F_m}, M_m: {self.M_m}')
        return super().dynamics(t, X, U)
    

if __name__ == "__main__":
    quad = Quad()
    quad.set_position(np.array([0, 0, -10]))

    T_sim = 5.0  # simulation time in seconds
    dt = 0.01  # time step
    num_steps = int(T_sim / dt)
    quad.set_dt(dt)
    quad.init_solver(quad.dynamics, type='rk4')
    
    X_data = []
    t_data = []
    for i in range(num_steps):
        quad.dynamics_step()
        X_data.append(quad.get_state())
        t_data.append(i * dt)

    from matplotlib import pyplot as plt

    X_data = np.array(X_data)
    vel = X_data[:, quad.vel_idx]

    acc = np.zeros_like(vel)
    acc[1:] = (vel[1:] - vel[:-1]) / dt
    acc[0] = acc[1]

    acc[acc < 1e-8] = 0

    # Plot positions, velocities, accelerations in columns
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs[i,0].plot(t_data, X_data[:, i], label=f'{labels[i]}')
        axs[i,0].set_ylabel(f'{labels[i]} [m]')
        axs[i,0].legend()
        axs[i,0].grid(True)

        axs[i,1].plot(t_data, vel[:, i], label=f'V{labels[i]}')
        axs[i,1].set_ylabel(f'V{labels[i]} [m/s]')
        axs[i,1].legend()
        axs[i,1].grid(True)

        axs[i,2].plot(t_data, acc[:, i], label=f'A{labels[i]}')
        axs[i,2].set_ylabel(f'A{labels[i]} [m/s²]')
        axs[i,2].legend()
        axs[i,2].grid(True)

    axs[2,0].set_xlabel('Time (s)')
    axs[2,1].set_xlabel('Time (s)')
    axs[2,2].set_xlabel('Time (s)')
    fig.suptitle('Quad Position, Velocity, and Acceleration Over Time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()