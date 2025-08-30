import numpy as np
import matplotlib.pyplot as plt

# m \ddot{x} &= -(u_1 + u_2)\sin\theta \\
# m \ddot{z} &= mg -(u_1 + u_2)\cos\theta \\
# I \ddot{\theta} &= r (u_1 - u_2)

# T_z = -(u1 + u2)  # total thrust/ minus since motors upwards, z downwards
# M_y = r * (u1 - u2)  # torque, u1 is forward motor, u2 is backward motor


m = 1.0  # mass of the quadrotor
I_yy = 0.1  # moment of inertia
G = 9.81  # gravitational acceleration
r = 0.5  # half-distance between rotors

T_sim = 10.0  # total simulation time
dt = 0.01  # time step for simulation
num_steps = int(T_sim / dt)
time_steps = np.arange(0, T_sim, dt)

x = 0.0  # initial x position
z = -10.0  # initial y position
theta = 0.0  # initial angle
x_dot = 0.0  # initial x velocity
z_dot = 0.0  # initial y velocity
theta_dot = 0.0  # initial angular velocity
theta_dot_prev = 0.0  # previous angular velocity for calculating acceleration

# State estimation variables
theta_est = 0.0  # estimated angle
x_est = 0.0      # estimated x position
z_est = -10.0    # estimated z position (same as initial z)
x_dot_est = 0.0  # estimated x velocity
z_dot_est = 0.0  # estimated z velocity

# simulate physics using Euler integration
states = np.zeros((num_steps, 6))  # state vector: [x, z, theta, x_dot, z_dot, theta_dot]
states_est = np.zeros((num_steps, 6))  # estimated state 

states[0] = [x, z, theta, x_dot, z_dot, theta_dot]
states_est[0] = [x_est, z_est, theta_est, x_dot_est, z_dot_est, theta_dot]
w_sp_history = np.zeros(num_steps)
u_history = np.zeros((num_steps, 2))
theta_dot_dot_history = np.zeros(num_steps)
theta_sp_history = np.zeros(num_steps)
x_dot_sp_history = np.zeros(num_steps)
z_dot_sp_history = np.zeros(num_steps)
a_x_sp_history = np.zeros(num_steps)
a_z_sp_history = np.zeros(num_steps)
x_ddot_history = np.zeros(num_steps)
z_ddot_history = np.zeros(num_steps)

theta_dot_sp = 0.0  # desired angular velocity (for PD control)
theta_sp = 0.0  # desired angle (not used in this simulation)
K_p_theta_dot = 0.2
K_p_theta = 2
K_d = 1.0 # PD control gains

x_dot_sp = 0.0  # desired horizontal velocity
z_dot_sp = 0.0  # desired vertical velocity
a_x_sp = 0.0
a_z_sp = 0.0
x_sp = 0.0  # desired horizontal position setpoint
z_sp = 0.0  # desired vertical position setpoint

x_dot_sp = 0.0  # desired horizontal velocity setpoint
z_dot_sp = 0.0  # desired vertical velocity setpoint

T_z_sp = m * G 
M_y_sp = 0.0  # desired torque (not used in this simulation)


#TODO TRY ALL MODES and see how they work
modes = ['crash', 'acro', 'stab', 'acceleration', 'velocity', 'position']
mode = "stab"
# mode = "position"
# mode = "position"
# mode = modes[1]
# mode = modes[2]
# mode = modes[3]
# mode = modes[4]
# mode = modes[5]

# Control options
use_estimated_position = True  # Set to True to use estimated position in position control
# use_estimated_position = False  # Set to True to use estimated position in position control

### STATE ESTIMATION

# Kalman filter variables for theta estimation
theta_kf = 0.0 # Kalman filter state (estimated angle)
P_kf = 1.01      # Error covariance
Q_kf = 0.01     # Process noise (gyroscope drift/bias)
R_kf = 200.5      # Measurement noise (accelerometer noise)

def estimate_theta(a_x, a_z):
    # use accelerometer data to estimate theta
    theta_imu = -np.arctan2(a_x, -a_z)
    return theta_imu


def estimate_theta_kalman(a_x, a_z, theta_dot):
    global theta_kf, P_kf, Q_kf, R_kf, dt
    # Predict
    theta_kf = theta_kf + theta_dot * dt
    P_kf = P_kf + Q_kf

    # Measurement update
    theta_meas = -np.arctan2(a_x, -a_z)
    K = P_kf / (P_kf + R_kf)  # Kalman gain
    theta_kf = theta_kf + K * (theta_meas - theta_kf)
    P_kf = (1 - K) * P_kf
    return theta_kf


def estimate_pos(a_x, a_z, theta):
    global x_est, z_est, x_dot_est, z_dot_est
    
    # first rotate local to global accelerations
    R_LW = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    a_global = R_LW @ np.array([a_x, a_z])
    a_x_est, a_z_est = a_global[0], a_global[1]

    # Simple double integration
    # First integration: acceleration -> velocity
    x_dot_est += a_x_est * dt
    z_dot_est += a_z_est * dt

    # Second integration: velocity -> position
    x_est += x_dot_est * dt
    z_est += z_dot_est * dt
    return x_est, z_est

def accelerometer(a_x_gt, a_z_gt, theta_gt, noise=0.1):
    # input ground truth accelerations from physics sim
    # rotation matrix from world to local
    R_WL = np.array([[np.cos(theta_gt), np.sin(theta_gt)],
                  [-np.sin(theta_gt), np.cos(theta_gt)]])
    
    a_local = R_WL @ np.array([a_x_gt, a_z_gt - G])  # substract gravity to z component
    a_x = a_local[0] + np.random.normal(0, noise)
    a_z = a_local[1] + np.random.normal(0, noise)
    return a_x, a_z

def gyroscope(theta_dot_gt, noise=0.01):
    theta_dot = theta_dot_gt + np.random.normal(0, noise)
    return theta_dot

## CONTROL

def crash_control(T_z_sp, M_y_sp):
    return T_z_sp, M_y_sp

def rate_control(T_z_sp, theta_dot_sp):
    M_y_sp = K_p_theta_dot * (theta_dot_sp - theta_dot)
    return T_z_sp, M_y_sp

def att_control(T_z_sp, theta_sp):
    global theta_dot_sp
    # theta_dot_sp = K_p_theta * (theta_sp - theta)
    theta_dot_sp = K_p_theta * (theta_sp - theta_est)
    # theta_dot_sp = K_p_theta * (theta_sp - theta_est)
    return rate_control(T_z_sp, theta_dot_sp)

def acceleration_control(a_x_sp, a_z_sp):
    global theta_sp, T_z_sp
    F_x_sp = m * a_x_sp
    F_z_sp = m * (G - a_z_sp)
    T_z_sp = np.sqrt(F_x_sp**2 + F_z_sp**2)
    theta_sp = np.arctan2(-F_x_sp, F_z_sp)
    return att_control(T_z_sp, theta_sp)

def velocity_control(x_dot_sp, z_dot_sp):
    global a_x_sp, a_z_sp
    # Calculate desired accelerations based on velocity setpoints
    error_x = x_dot_sp - x_dot
    error_z = z_dot_sp - z_dot
    k_p = 1.0  # proportional gain for velocity control
    a_x_sp = k_p * error_x
    a_z_sp = k_p * error_z
    return acceleration_control(a_x_sp, a_z_sp)

def position_control(x_sp, z_sp, use_estimated=False):
    global x_dot_sp, z_dot_sp
    
    # Choose between ground truth and estimated position
    if use_estimated:
        current_x = x_est
        current_z = z_est
    else:
        current_x = x
        current_z = z
    
    error_x = x_sp - current_x
    error_z = z_sp - current_z
    
    k_p_x = 0.3  # horizontal gain
    k_p_z = 0.5  # vertical gain (often different due to gravity)
    
    x_dot_sp = k_p_x * error_x
    z_dot_sp = k_p_z * error_z
    
    return velocity_control(x_dot_sp, z_dot_sp)

for i in range(1, num_steps):
    if i > num_steps // 2:
        if mode == 'crash':
            T_z_sp = m * G + 0.1  # crash mode, extra thrust to simulate crash
            M_y_sp = 0.001  # no torque in crash mode
        elif mode == 'acro':
            theta_dot_sp = 0.1  # desired angular velocity for acro mode
        elif mode == 'stab':
            theta_sp = 0.02  # desired angle for step response
        if mode == 'acceleration':
            a_x_sp = 0.1
            a_z_sp = -0.4
        if mode == 'velocity':
            x_dot_sp = 0.1
            z_dot_sp = -0.4
        if mode == 'position':
            x_sp = 5
            z_sp = -30

    if mode == 'position':
        T_z_sp, M_y_sp = position_control(x_sp, z_sp, use_estimated_position)
    elif mode == 'velocity':
        T_z_sp, M_y_sp = velocity_control(x_dot_sp, z_dot_sp)
    elif mode == 'acceleration':
        T_z_sp, M_y_sp = acceleration_control(a_x_sp, a_z_sp)
    elif mode == 'stab':
        T_z_sp, M_y_sp = att_control(T_z_sp, theta_sp)
    elif mode == 'acro':
        T_z_sp, M_y_sp = rate_control(T_z_sp, theta_dot_sp)

    # control allocation, solve linear system for u1 and u2
    u1, u2 = np.linalg.solve(np.array([[-1, -1], [r, -r]]), np.array([-T_z_sp, M_y_sp]))

    # compute accelerations
    x_ddot = -(u1 + u2) * np.sin(theta) / m
    z_ddot = G - (u1 + u2) * np.cos(theta) / m
    theta_ddot = r * (u1 - u2) / I_yy

    x_ddot_history[i] = x_ddot
    z_ddot_history[i] = z_ddot

    # update velocities
    x_dot += x_ddot * dt
    z_dot += z_ddot * dt
    theta_dot += theta_ddot * dt

    # update positions
    x += x_dot * dt
    z += z_dot * dt
    theta += theta_dot * dt

    a_x_imu, a_z_imu = accelerometer(x_ddot, z_ddot, theta, noise=0.1)
    theta_dot_imu = gyroscope(theta_dot, noise=0.01)
    
    # State estimation
    # theta_est =  estimate_theta(a_x_imu, a_z_imu)   
    theta_est = estimate_theta_kalman(a_x_imu, a_z_imu, theta_dot_imu)   
    x_est, z_est = estimate_pos(a_x_imu, a_z_imu, theta_est)  # estimate position using double integration

    # add ground constraint to z 
    z = min(z, 0.0)  # prevent going below ground level

    # data recording
    states[i] = [x, z, theta, x_dot, z_dot, theta_dot]
    states_est[i] = [x_est, z_est, theta_est, x_dot_est, z_dot_est, theta_dot]
    w_sp_history[i] = theta_dot_sp
    u_history[i] = [u1, u2]
    theta_sp_history[i] = theta_sp
    x_dot_sp_history[i] = x_dot_sp
    z_dot_sp_history[i] = z_dot_sp
    a_x_sp_history[i] = a_x_sp
    a_z_sp_history[i] = a_z_sp

# plot the results
fig, ax = plt.subplots(5, 2, figsize=(12, 16))
fig.tight_layout(pad=3.0)

ax[0, 0].plot(time_steps, states[:, 0], label='x position (GT)', color='blue')
ax[0, 0].plot(time_steps, states_est[:, 0], label='x position (EST)', color='cyan')
ax[0, 0].set_ylabel('x position (m)')
ax[0, 0].legend()

ax[0, 1].plot(time_steps, states[:, 1], label='z position (GT)', color='orange')
ax[0, 1].plot(time_steps, states_est[:, 1], label='z position (EST)', color='red')
ax[0, 1].set_ylabel('z position (m)')
ax[0, 1].legend()  

ax[1, 0].plot(time_steps, states[:, 3], label='x_dot velocity (GT)', color='magenta')
ax[1, 0].plot(time_steps, states_est[:, 3], label='x_dot velocity (EST)', color='pink')
ax[1, 0].plot(time_steps, x_dot_sp_history, label='x_dot setpoint', linestyle='--', color='red')
ax[1, 0].set_ylabel('x_dot (m/s)')
ax[1, 0].legend()

ax[1, 1].plot(time_steps, states[:, 4], label='z_dot velocity (GT)', color='brown')
ax[1, 1].plot(time_steps, states_est[:, 4], label='z_dot velocity (EST)', color='yellow')
ax[1, 1].plot(time_steps, z_dot_sp_history, label='z_dot setpoint', linestyle='--', color='red')
ax[1, 1].set_ylabel('z_dot (m/s)')
ax[1, 1].legend()

ax[2, 0].plot(time_steps, x_ddot_history, label='x_ddot', color='cyan')
ax[2, 0].plot(time_steps, a_x_sp_history, label='x_ddot setpoint', linestyle='--', color='red')
ax[2, 0].set_ylabel('x_ddot (m/s^2)')
ax[2, 0].legend()

ax[2, 1].plot(time_steps, z_ddot_history, label='z_ddot', color='lime')
ax[2, 1].plot(time_steps, a_z_sp_history, label='z_ddot setpoint', linestyle='--', color='red')
ax[2, 1].set_ylabel('z_ddot (m/s^2)')
ax[2, 1].legend()

ax[3, 0].plot(time_steps, states[:, 2], label='theta angle (GT)', color='green')
ax[3, 0].plot(time_steps, states_est[:, 2], label='theta angle (EST)', color='orange')
ax[3, 0].plot(time_steps, theta_sp_history, label='theta setpoint', linestyle='--', color='red')
ax[3, 0].set_ylabel('theta angle (rad)')
ax[3, 0].legend()

ax[3, 1].plot(time_steps, states[:, 5], label='angular velocity', color='purple')
ax[3, 1].plot(time_steps, w_sp_history, label='setpoint', linestyle='--', color='red')
ax[3, 1].set_ylabel('theta_dot (rad/s)')
ax[3, 1].legend()

ax[4, 0].plot(time_steps[1:], u_history[1:, 0], label='u1', color='blue')
ax[4, 0].plot(time_steps[1:], u_history[1:, 1], label='u2', color='cyan')
ax[4, 0].set_ylabel('Control inputs')
ax[4, 0].set_xlabel('Time (s)')
ax[4, 0].legend()

ax[4, 1].axis('off')

plt.show()

