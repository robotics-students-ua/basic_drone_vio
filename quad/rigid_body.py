# Rigid body class. Contains state vector, dynamics, solver, etc.
# It uses NED coordinate system (z axis is down) and quaternion representation of attitude




import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.integrate import RK45




def multiply_quaternions(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([x, y, z, w])


def quat_derivative(q, w):
    if w[0] == 0 and w[1] == 0 and w[2] == 0:
            q_d = np.zeros(4)
    else:
        q_d = 1/2 * multiply_quaternions(q, np.array([w[0], w[1], w[2], 0]))
    return q_d

def skew_w(w):
    return np.array([[0, -w[0], -w[1], -w[2]],
                    [w[0], 0, w[2], -w[1]],
                    [w[1], -w[2], 0, w[0]],
                    [w[2], w[1], -w[0], 0]])



class RigidBody:
    G = 9.81

    def __init__(self) -> None:
        self.X = np.zeros(13) # state vector (p,v,q,w)
        self.pos_idx = [0, 1, 2]
        self.vel_idx = [3, 4, 5]
        self.quaternion_idx = [6, 7, 8, 9]
        self.ang_vel_idx = [10, 11, 12]

        # physical params. Redefine in vehicle child class
        self.M = 1
        self.I = np.eye(3)
        self.I_inv = np.linalg.inv(self.I)

        # simulation params
        self.dt = 0.04
        self.v_d = 0 # acceleration/ required for accelerometer
        self.solver = None

        self.aero = None

        # aerodynamics forces and moments. Defined in vehicle child class
        self.F_a = np.zeros(3)
        self.M_a = np.zeros(3)
        # thrust and moment from motors. Defined in vehicle child class
        self.F_m = np.zeros(3)
        self.M_m = np.zeros(3)

    @property 
    def p(self): return self.X[self.pos_idx]
    
    @property
    def v(self): return self.X[self.vel_idx]
    
    @property
    def q(self): return self.X[self.quaternion_idx]
    
    @property
    def w(self): return self.X[self.ang_vel_idx]
    
    @property
    def euler_angles(self): return R.from_quat(self.q).as_euler('xyz')

    def get_state(self):
        return self.X

    def set_position(self, p):
        self.X[self.pos_idx] = p

    def set_velocity(self, v):
        self.X[self.vel_idx] = v

    def set_orientation(self, q):
        self.X[self.quaternion_idx] = q

    def set_mass(self, m):
        self.M = m

    def set_inertia(self, I):
        self.I = I

    def set_initial_state(self, X):
        self.X = X
    
    def set_dt(self, dt):
        self.dt = dt


    def init_solver(self, dynamics_function, type='rk4'):
        if type == 'rk4':
            self.solver = RK45(dynamics_function, 0, self.X, np.inf, self.dt)
        if type == 'euler':
            self.solver = self.euler_solver()

    

    def dynamics(self, t, X, U=None):
        if self.aero is None:
            # self.F_a = np.zeros(3)
            # self.M_a = np.zeros(3)
            # first order drag and dumper
            k_dv = 0.1
            k_dw = 0.025
            self.F_a = - k_dv * R.from_quat(self.q).inv().apply(self.v)
            self.M_a = - k_dw * self.w

        p_d = self.v
        self.v_d = R.from_quat(self.q).apply(self.F_a + self.F_m) / self.M + np.array([0, 0, self.G])
        q_d = quat_derivative(self.q, self.w)
        w_d = self.I_inv @ (self.M_a + self.M_m - np.cross(self.w, self.I @ self.w))
        return np.hstack((p_d, self.v_d, q_d, w_d))

    def euler_solver(self):
        def step():
            self.X += self.dynamics() * self.dt
            pass
        pass


    def on_ground(self):
        # On ground if z >= 0 (NED: z down, so ground is at z=0)
        return self.p[2] >= 0

    def dynamics_step(self):
        restitution = 0.8  # 1.0 = perfectly elastic, <1 = some energy loss
        # Step the solver first
        self.solver.step()
        self.X = self.solver.y
        # Check for collision with ground
        if self.p[2] >= 0:
            # Set position to ground level
            self.X[self.pos_idx[2]] = 0
            # If moving downward, bounce
            if self.v[2] > 0:
                self.X[self.vel_idx[2]] = -self.v[2] * restitution
                # Optionally, dampen horizontal velocity a bit (friction)
                self.X[self.vel_idx[0]] *= 0.95
                self.X[self.vel_idx[1]] *= 0.95
            else:
                # If already moving up or stationary, just stay on ground
                self.X[self.vel_idx[2]] = 0
            # Optionally, dampen angular speed on collision
            self.X[self.ang_vel_idx] *= 0.8



if __name__ == "__main__":
    quad = RigidBody()
    quad.set_mass(1.0)
    quad.set_inertia(np.diag([0.1, 0.1, 0.1]))
    X = np.zeros(13)
    X[quad.quaternion_idx] = R.from_euler('xyz', [0, 0, 0]).as_quat()
    X[quad.pos_idx] = np.array([0, 0, -50])
    quad.set_initial_state(X)
    quad.set_dt(0.04)
    quad.init_solver(quad.dynamics, type='rk4')

    # Run simulation
    X_history = []
    T_sim = 20
    for t in np.arange(0, T_sim, quad.dt):
        quad.dynamics_step()
        X_history.append(quad.X.copy())

    # Plot results
    import matplotlib.pyplot as plt
    X_history = np.array(X_history)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(X_history[:, quad.pos_idx[0]], X_history[:, quad.pos_idx[1]], X_history[:, quad.pos_idx[2]])
    plt.title("Position")
    plt.subplot(3, 1, 2)
    plt.plot(X_history[:, quad.vel_idx[0]], X_history[:, quad.vel_idx[1]], X_history[:, quad.vel_idx[2]])
    plt.title("Velocity")
    plt.subplot(3, 1, 3)
    plt.plot(X_history[:, quad.quaternion_idx[0]], X_history[:, quad.quaternion_idx[1]], X_history[:, quad.quaternion_idx[2]], X_history[:, quad.quaternion_idx[3]])
    plt.title("Orientation")
    plt.tight_layout()
    plt.show()