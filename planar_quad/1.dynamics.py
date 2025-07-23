import numpy as np
import matplotlib.pyplot as plt

# m \ddot{x} &= -(u_1 + u_2)\sin\theta \\
# m \ddot{z} &= mg -(u_1 + u_2)\cos\theta \\
# I \ddot{\theta} &= r (u_1 - u_2)

m = 1.0  # mass of the quadrotor
I_yy = 0.1  # moment of inertia
g = 9.81  # gravitational acceleration
r = 0.5  # half-distance between rotors

T_sim = 10.0  # total simulation time
dt = 0.01  # time step for simulation
num_steps = int(T_sim / dt)
time_steps = np.arange(0, T_sim, dt)

x_0 = 0.0  # initial x position
z_0 = 0.0  # initial y position
theta_0 = 0.0  # initial angle
x_dot_0 = 0.0  # initial x velocity
z_dot_0 = 0.0  # initial y velocity
theta_dot_0 = 0.0  # initial angular velocity

# simulate physics using Euler integration
states = np.zeros((num_steps, 6))  # state vector: [x, z, theta, x_dot, z_dot, theta_dot]
states[0] = [x_0, z_0, theta_0, x_dot_0, z_dot_0, theta_dot_0]      

for i in range(1, num_steps):
    # extract current state
    x, z, theta, x_dot, z_dot, theta_dot = states[i - 1]

    T_z = m * g + 0.5 # total thrust to balance gravity plus some extra force
    M_y = 0.001  # small torque for testing

    # T_z = -(u1 + u2)  # total thrust/ minus since motors upwards, z downwards
    # M_y = r * (u1 - u2)  # torque, u1 is forward motor, u2 is backward motor

    # control allocation, solve linear system for u1 and u2
    u1, u2 = np.linalg.solve(np.array([[-1, -1], [r, -r]]), np.array([-T_z, M_y]))
    print(f"Step {i}: u1 = {u1}, u2 = {u2}")

    # compute accelerations
    x_ddot = -(u1 + u2) * np.sin(theta) / m
    z_ddot = g - (u1 + u2) * np.cos(theta) / m
    theta_ddot = r * (u1 - u2) / I_yy

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

    # store new state
    states[i] = [x, z, theta, x_dot, z_dot, theta_dot]

# plot the results
fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].plot(time_steps, states[:, 0], label='x position')
ax[0].set_ylabel('x position (m)')
ax[0].legend()
ax[1].plot(time_steps, states[:, 1], label='z position', color='orange')
ax[1].set_ylabel('z position (m)')
ax[1].legend()  
ax[2].plot(time_steps, states[:, 2], label='theta angle', color='green')
ax[2].set_ylabel('theta angle (rad)')
ax[2].set_xlabel('Time (s)')
ax[2].legend()
ax[2].set_ylim(-np.pi, np.pi)  # limit theta to [-pi, pi]
plt.show()

