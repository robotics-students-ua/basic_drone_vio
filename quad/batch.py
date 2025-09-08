import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from quad import Quad
from controlers import StabController

sim_time = 10.0  # seconds
dt = 0.01       # simulation timestep
steps = int(sim_time / dt)

vehicle = Quad()
vehicle.set_dt(dt)
vehicle.init_solver(vehicle.dynamics, type='rk4')
controller = StabController()

# Initial state
vehicle.set_position(np.array([0, 0, -100]))
thrust_z = -vehicle.M * 9.81
roll_sp = 0.0
pitch_sp = 0.0
yaw_sp = 0.0

# Data logging
log_t = []
log_pos = []
log_euler = []
log_sp = []
log_u = []

for i in range(steps):
    t = i * dt
    # Setpoint: step in pitch between 2s and 4s
    if 2.0 <= t < 4.0:
        pitch_sp = 0.2
        # roll_sp = 0.2
        # yaw_sp = 0.2
    else:
        pitch_sp = 0.0
        # roll_sp = 0.0
        # yaw_sp = 0.0
    q_sp = R.from_euler('xyz', [roll_sp, pitch_sp, yaw_sp]).as_quat()
    q = vehicle.q  # current quaternion
    w = vehicle.w  # current angular velocity

    u = controller.compute_control(
        q_sp=[q_sp[3], q_sp[0], q_sp[1], q_sp[2]],  # [w, x, y, z]
        q=[q[3], q[0], q[1], q[2]],                # [w, x, y, z]
        w=w,
        thrust_z=thrust_z
    )
    vehicle.update_controls(u)
    vehicle.dynamics_step()

    # Log data
    log_t.append(t)
    log_pos.append(vehicle.p.copy())
    log_euler.append(R.from_quat(vehicle.q).as_euler('xyz', degrees=True))
    log_sp.append([roll_sp, pitch_sp, yaw_sp])
    log_u.append(u)

# Convert logs to arrays
log_t = np.array(log_t)
log_pos = np.array(log_pos)
log_euler = np.array(log_euler)
log_sp = np.array(log_sp)
log_u = np.array(log_u)

# Calculate velocities from positions (finite difference)
log_vel = np.zeros_like(log_pos)
log_vel[1:] = (log_pos[1:] - log_pos[:-1]) / dt
log_vel[0] = log_vel[1]  # repeat first value for shape

# Calculate accelerations from velocities (finite difference)
log_acc = np.zeros_like(log_vel)
log_acc[1:] = (log_vel[1:] - log_vel[:-1]) / dt
log_acc[0] = log_acc[1]  # repeat first value for shape

#zeroing
log_euler[np.abs(log_euler) < 1e-8] = 0
log_pos[np.abs(log_pos) < 1e-8] = 0
log_vel[np.abs(log_vel) < 1e-8] = 0
log_acc[np.abs(log_acc) < 1e-8] = 0

# Plot using subfigures: 4 columns (angles, positions, velocities, accelerations), 3 rows (X, Y, Z)
fig, axs = plt.subplots(3, 4, figsize=(20, 10), sharex=True)

# Row 0: Roll, X, Vx, Ax
axs[0,0].plot(log_t, log_euler[:,0], label='Roll')
axs[0,0].plot(log_t, np.rad2deg(log_sp[:,0]), '--', label='Roll SP')
axs[0,0].set_ylabel('Roll [deg]')
axs[0,0].legend()
axs[0,0].grid(True)

axs[0,1].plot(log_t, log_pos[:,0], label='X')
axs[0,1].set_ylabel('X [m]')
axs[0,1].legend()
axs[0,1].grid(True)

axs[0,2].plot(log_t, log_vel[:,0], label='Vx')
axs[0,2].set_ylabel('Vx [m/s]')
axs[0,2].legend()
axs[0,2].grid(True)

axs[0,3].plot(log_t, log_acc[:,0], label='Ax')
axs[0,3].set_ylabel('Ax [m/s²]')
axs[0,3].legend()
axs[0,3].grid(True)

# Row 1: Pitch, Y, Vy, Ay
axs[1,0].plot(log_t, log_euler[:,1], label='Pitch')
axs[1,0].plot(log_t, np.rad2deg(log_sp[:,1]), '--', label='Pitch SP')
axs[1,0].set_ylabel('Pitch [deg]')
axs[1,0].legend()
axs[1,0].grid(True)

axs[1,1].plot(log_t, log_pos[:,1], label='Y')
axs[1,1].set_ylabel('Y [m]')
axs[1,1].legend()
axs[1,1].grid(True)

axs[1,2].plot(log_t, log_vel[:,1], label='Vy')
axs[1,2].set_ylabel('Vy [m/s]')
axs[1,2].legend()
axs[1,2].grid(True)

axs[1,3].plot(log_t, log_acc[:,1], label='Ay')
axs[1,3].set_ylabel('Ay [m/s²]')
axs[1,3].legend()
axs[1,3].grid(True)

# Row 2: Yaw, Z, Vz, Az
axs[2,0].plot(log_t, log_euler[:,2], label='Yaw')
axs[2,0].plot(log_t, np.rad2deg(log_sp[:,2]), '--', label='Yaw SP')
axs[2,0].set_ylabel('Yaw [deg]')
axs[2,0].set_xlabel('Time [s]')
axs[2,0].legend()
axs[2,0].grid(True)

axs[2,1].plot(log_t, log_pos[:,2], label='Z')
axs[2,1].set_ylabel('Z [m]')
axs[2,1].set_xlabel('Time [s]')
axs[2,1].legend()
axs[2,1].grid(True)

axs[2,2].plot(log_t, log_vel[:,2], label='Vz')
axs[2,2].set_ylabel('Vz [m/s]')
axs[2,2].set_xlabel('Time [s]')
axs[2,2].legend()
axs[2,2].grid(True)

axs[2,3].plot(log_t, log_acc[:,2], label='Az')
axs[2,3].set_ylabel('Az [m/s²]')
axs[2,3].set_xlabel('Time [s]')
axs[2,3].legend()
axs[2,3].grid(True)

fig.suptitle('Quadrotor Simulation: Angles, Positions, Velocities, Accelerations')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()