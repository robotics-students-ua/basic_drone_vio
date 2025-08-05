import numpy as np
import matplotlib.pyplot as plt

# m \ddot{x} &= -(u_1 + u_2)\sin\theta \\
# m \ddot{z} &= mg -(u_1 + u_2)\cos\theta \\
# I \ddot{\theta} &= r (u_1 - u_2)

# T_z = -(u1 + u2)  # total thrust/ minus since motors upwards, z downwards
# M_y = r * (u1 - u2)  # torque, u1 is forward motor, u2 is backward motor


m = 1.0  # mass of the quadrotor
I_yy = 0.1  # moment of inertia
g = 9.81  # gravitational acceleration
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

# simulate physics using Euler integration
states = np.zeros((num_steps, 6))  # state vector: [x, z, theta, x_dot, z_dot, theta_dot]
states[0] = [x, z, theta, x_dot, z_dot, theta_dot]
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

x_dot_sp = 0.0  # desired horizontal velocity setpoint
z_dot_sp = 0.0  # desired vertical velocity setpoint

T_z_sp = m * g 
M_y_sp = 0.0  # desired torque (not used in this simulation)


#TODO TRY ALL MODES and see how they work
modes = ['crash', 'acro', 'stab', 'acceleration', 'velocity']
mode = modes[0]
mode = modes[1]
mode = modes[2]
mode = modes[3]
mode = modes[4]

def crash_control(T_z_sp, M_y_sp):
    return T_z_sp, M_y_sp

def rate_control(T_z_sp, theta_dot_sp):
    M_y_sp = K_p_theta_dot * (theta_dot_sp - theta_dot)
    return T_z_sp, M_y_sp

def att_control(T_z_sp, theta_sp):
    global theta_dot_sp
    theta_dot_sp = K_p_theta * (theta_sp - theta)
    return rate_control(T_z_sp, theta_dot_sp)

def acceleration_control(a_x_sp, a_z_sp):
    global theta_sp, T_z_sp
    F_x_sp = m * a_x_sp
    F_z_sp = m * (g - a_z_sp)
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

for i in range(1, num_steps):
    if i > num_steps // 2:
        if mode == 'crash':
            T_z_sp = m * g + 0.1  # crash mode, extra thrust to simulate crash
            M_y_sp = 0.001  # no torque in crash mode
        elif mode == 'acro':
            theta_dot_sp = 0.1  # desired angular velocity for acro mode
        elif mode == 'stab':
            theta_sp = 0.1  # desired angle for step response
        if mode == 'acceleration':
            a_x_sp = 0.1
            a_z_sp = -0.4
        if mode == 'velocity':
            x_dot_sp = 0.1
            z_dot_sp = -0.4

    if mode == 'velocity':
        T_z_sp, M_y_sp = velocity_control(x_dot_sp, z_dot_sp)
    if mode == 'acceleration':
        T_z_sp, M_y_sp = acceleration_control(a_x_sp, a_z_sp)
    if mode == 'stab':
        T_z_sp, M_y_sp = att_control(T_z_sp, theta_sp)
    if mode == 'acro':
        T_z_sp, M_y_sp = rate_control(T_z_sp, theta_dot_sp)

    # control allocation, solve linear system for u1 and u2
    u1, u2 = np.linalg.solve(np.array([[-1, -1], [r, -r]]), np.array([-T_z_sp, M_y_sp]))

    # compute accelerations
    x_ddot = -(u1 + u2) * np.sin(theta) / m
    z_ddot = g - (u1 + u2) * np.cos(theta) / m
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

    # add ground constraint to z 
    z = min(z, 0.0)  # prevent going below ground level

    # data recording
    states[i] = [x, z, theta, x_dot, z_dot, theta_dot]
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

ax[0, 0].plot(time_steps, states[:, 0], label='x position')
ax[0, 0].set_ylabel('x position (m)')
ax[0, 0].legend()

ax[0, 1].plot(time_steps, states[:, 1], label='z position', color='orange')
ax[0, 1].set_ylabel('z position (m)')
ax[0, 1].legend()  

ax[1, 0].plot(time_steps, states[:, 3], label='x_dot velocity', color='magenta')
ax[1, 0].plot(time_steps, x_dot_sp_history, label='x_dot setpoint', linestyle='--', color='red')
ax[1, 0].set_ylabel('x_dot (m/s)')
ax[1, 0].legend()

ax[1, 1].plot(time_steps, states[:, 4], label='z_dot velocity', color='brown')
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

ax[3, 0].plot(time_steps, states[:, 2], label='theta angle', color='green')
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

