import numpy as np
import matplotlib.pyplot as plt
import cv2

# m \ddot{x} &= -(u_1 + u_2)\sin\theta \\
# m \ddot{z} &= mg -(u_1 + u_2)\cos\theta \\
# I \ddot{\theta} &= r (u_1 - u_2)

m = 1.0  # mass of the quadrotor
I_yy = 0.1  # moment of inertia
g = 9.81  # gravitational acceleration
r = 0.5  # half-distance between rotors

dt = 0.01  # time step for simulation
max_sim_time = 300.0  # maximum simulation time

# initial conditions
x = 0.0  # initial x position
z = -10.0  # initial z position (start above ground)
theta = 0.0  # initial angle
x_dot = 0.0  # initial x velocity
z_dot = 0.0  # initial z velocity
theta_dot = 0.0  # initial angular velocity

# simulation bounds
z_min = -100  # minimum z position to prevent going below ground level
z_max = 0.0  # maximum z position (ground level)
x_min = -100  # minimum x position
x_max = 100  # maximum x position

# visualization parameters
window_width = 1200
window_height = 600
scale = 5  # pixels per meter
center_x = window_width // 2
# center_y = window_height // 2
center_y = int(window_height * 0.9)  # center of the window in y-axis

# for storing trajectory
trajectory = []
current_time = 0.0

# control inputs (can be modified by keyboard)
thrust_offset = 0.0  # additional thrust beyond gravity compensation
torque_input = 0.0  # torque input
thrust_step = 0.1  # how much thrust changes per key press
torque_step = 0.001  # how much torque changes per key press

a_step = 1
v_step = 1
x_step = 1
z_step = 1

w_sp_step = 0.01  # step size for angular velocity setpoint
theta_step = 0.1  # step size for angle setpoint
# create OpenCV window
cv2.namedWindow('Drone Simulation', cv2.WINDOW_AUTOSIZE)


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
z_sp = -10.0  # desired vertical position setpoint

x_dot_sp = 0.0  # desired horizontal velocity setpoint
z_dot_sp = 0.0  # desired vertical velocity setpoint

T_z_sp = m * g 
M_y_sp = 0.0  # desired torque (not used in this simulation)

#TODO Add position mode
modes = ['crash', 'acro', 'stab', 'acceleration', 'velocity', 'position']
mode_index = 4
mode = modes[mode_index]

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


def position_control(x_sp, z_sp):
    global x_dot_sp, z_dot_sp
    error_x = x_sp - x
    error_z = z_sp - z
    
    k_p_x = 0.3  # horizontal gain
    k_p_z = 0.5  # vertical gain (often different due to gravity)
    
    x_dot_sp = k_p_x * error_x
    z_dot_sp = k_p_z * error_z
    
    return velocity_control(x_dot_sp, z_dot_sp)

# simulation loop
while current_time < max_sim_time:
    # create blank image
    img = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    
    # draw coordinate system as arrows
    # x-axis arrow (pointing right)
    arrow_length = 30
    arrow_start_x = center_x
    arrow_start_y = center_y
    cv2.arrowedLine(img, (arrow_start_x, arrow_start_y), 
                   (arrow_start_x + arrow_length, arrow_start_y), 
                   (255, 255, 255), 2, tipLength=0.3)
    cv2.putText(img, "x", (arrow_start_x + arrow_length + 5, arrow_start_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # z-axis arrow (pointing down)
    cv2.arrowedLine(img, (arrow_start_x, arrow_start_y), 
                   (arrow_start_x, arrow_start_y + arrow_length), 
                   (255, 255, 255), 2, tipLength=0.3)
    cv2.putText(img, "z", (arrow_start_x + 5, arrow_start_y + arrow_length + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # draw simulation bounds
    # x bounds (vertical lines)
    x_min_screen = int(center_x + x_min * scale)
    x_max_screen = int(center_x + x_max * scale)
    cv2.line(img, (x_min_screen, 0), (x_min_screen, window_height), (255, 0, 0), 2)  # red left bound
    cv2.line(img, (x_max_screen, 0), (x_max_screen, window_height), (255, 0, 0), 2)  # red right bound
    
    # z bounds (horizontal lines)
    z_min_screen = int(center_y + z_min * scale)
    z_max_screen = int(center_y + z_max * scale)
    cv2.line(img, (0, z_min_screen), (window_width, z_min_screen), (0, 255, 0), 1)  # green top bound (z_min)
    cv2.line(img, (0, z_max_screen), (window_width, z_max_screen), (0, 255, 0), 1)  # green bottom bound (z_max)
    

    if mode == 'crash':
        # T_z_sp, M_y_sp = crash_control(T_z_sp, M_y_sp)
        # T_z_sp = T_z_sp  # total thrust with user control
        M_y_sp = torque_input  # total torque with user control
    elif mode == 'position':
        T_z_sp, M_y_sp = position_control(x_sp, z_sp)
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


    # T_z = m * g + thrust_offset  # total thrust with user control
    # M_y = torque_input  # torque with user control


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

    # add boundary effects with elastic collision
    # X boundaries
    if x <= x_min:
        x = x_min
        x_dot = -x_dot * 0.8  # reverse velocity with some energy loss
    elif x >= x_max:
        x = x_max
        x_dot = -x_dot * 0.8
    
    # Z boundaries
    if z <= z_min:
        z = z_min
        z_dot = -z_dot * 0.8  # reverse velocity with some energy loss
    elif z >= z_max:
        z = z_max
        z_dot = -z_dot * 0.8
    
    # store trajectory point
    trajectory.append((x, z))
    
    # convert drone position to screen coordinates
    screen_x = int(center_x + x * scale)
    screen_y = int(center_y + z * scale)  # flip y-axis (screen y increases downward)
    
    # draw trajectory
    for i in range(1, len(trajectory)):
        prev_x = int(center_x + trajectory[i-1][0] * scale)
        prev_y = int(center_y + trajectory[i-1][1] * scale)
        curr_x = int(center_x + trajectory[i][0] * scale)
        curr_y = int(center_y + trajectory[i][1] * scale)
        cv2.line(img, (prev_x, prev_y), (curr_x, curr_y), (0, 255, 255), 1)
    
    # draw drone body as a line
    drone_width = int(20 * r * scale)  # line length representing drone body
    
    # calculate rotated line endpoints
    cos_theta = np.cos(theta)
    sin_theta = -np.sin(theta)
    
    # line endpoints relative to center
    half_width = drone_width // 2
    x1 = int(screen_x - half_width * cos_theta)
    y1 = int(screen_y - half_width * sin_theta)
    x2 = int(screen_x + half_width * cos_theta)
    y2 = int(screen_y + half_width * sin_theta)
    
    # draw drone body as a thick line
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # draw single thrust vector arrow from center
    # Calculate total thrust direction (perpendicular to drone body, pointing "up" in drone frame)
    arrow_dir_x = sin_theta  # perpendicular to drone body
    arrow_dir_y = -cos_theta
    
    # Fixed arrow length
    arrow_length = 20  # fixed size in pixels
    
    # Calculate arrow endpoint
    arrow_end_x = int(screen_x + arrow_length * arrow_dir_x)
    arrow_end_y = int(screen_y + arrow_length * arrow_dir_y)
    
    # Draw thrust arrow from center
    arrow_color = (0, 255, 0) 
    cv2.arrowedLine(img, (screen_x, screen_y), (arrow_end_x, arrow_end_y), 
                       arrow_color, 3, tipLength=0.3)
    
    # add text information
    info_text = [
        f"Time: {current_time:.2f}s",
        f"Mode: {mode.capitalize()}",
        f"Position: ({x:.2f}, {z:.2f})",
        f"Velocity: ({x_dot:.2f}, {z_dot:.2f})",
        f"Angle: {theta:.3f} rad",
        f"Angular speed: {theta_dot:.3f} rad/s",
        f"Controls: u1={u1:.2f}, u2={u2:.2f}",
        # f"Thrust z: {T_z_sp:.2f}",
        # f"Torque sp: {torque_input:.3f}",
        "",
        "Controls:",
        "UP/DOWN: Thrust +/-",
        "LEFT/RIGHT: Torque +/-",
        "M: Switch Mode",
        "R: Reset simulation",
        "ESC: Exit"
    ]
    
    # add setpoints information (upper right)
    setpoints_text = [
        "Current Setpoints:",
        f"Position: x_sp={x_sp:.2f}, z_sp={z_sp:.2f}",
        f"Velocity: x_dot_sp={x_dot_sp:.2f}, z_dot_sp={z_dot_sp:.2f}",
        f"Acceleration: a_x_sp={a_x_sp:.2f}, a_z_sp={a_z_sp:.2f}",
        f"Attitude: theta_sp={theta_sp:.3f} rad",
        f"Angular vel: theta_dot_sp={theta_dot_sp:.3f} rad/s",
        f"Thrust: T_z_sp={T_z_sp:.2f}",
        f"Torque: M_y_sp={M_y_sp:.3f}"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(img, text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # display setpoints in upper right corner
    for i, text in enumerate(setpoints_text):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x_pos = window_width - text_size[0] - 10
        y_pos = 30 + i * 20
        cv2.putText(img, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # show image
    cv2.imshow('Drone Simulation', img)
    
    # handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('m'):
        mode_index = (mode_index + 1) % len(modes)
        mode = modes[mode_index]
    elif key == 82 or key == ord('w'):  # UP arrow or W key
        if mode == 'crash' or mode == 'acro' or mode == 'stab':
            # thrust_offset += thrust_step
            T_z_sp += thrust_step  # update total thrust
        elif mode == 'acceleration':
            a_z_sp -= a_step
        elif mode == 'velocity':
            z_dot_sp -= v_step
        elif mode == 'position':
            z_sp -= z_step

    elif key == 84 or key == ord('s'):  # DOWN arrow or S key
        if mode == 'crash' or mode == 'acro' or mode == 'stab':
            T_z_sp -= thrust_step  # update total thrust
        elif mode == 'acceleration':
            a_z_sp += a_step
        elif mode == 'velocity':
            z_dot_sp += v_step
        elif mode == 'position':
            z_sp += z_step

    elif key == 81 or key == ord('a'):  # LEFT arrow or A key
        if mode == 'crash':
            torque_input += torque_step
        elif mode == 'acro':
            theta_dot_sp += w_sp_step
        elif mode == 'stab':
            theta_sp += theta_step     
        elif mode == 'acceleration':
            a_x_sp -= a_step
        elif mode == 'velocity':
            x_dot_sp -= v_step
        elif mode == 'position':
            x_sp -= x_step
    elif key == 83 or key == ord('d'):  # RIGHT arrow or D key
        if mode == 'crash':
            torque_input -= torque_step
        elif mode == 'acro':
            theta_dot_sp -= w_sp_step
        elif mode == 'stab':
            theta_sp -= theta_step
        elif mode == 'acceleration':
            a_x_sp += a_step
        elif mode == 'velocity':
            x_dot_sp += v_step
        elif mode == 'position':
            x_sp += x_step
    elif key == ord('r') or key == ord('R'):  # R key to reset
        # reset drone state
        x = 0.0
        z = -10.0
        theta = 0.0
        x_dot = 0.0
        z_dot = 0.0
        theta_dot = 0.0
        # reset controls
        thrust_offset = 0.0
        torque_input = 0.0
        # clear trajectory
        trajectory = []
        # reset time
        current_time = 0.0
    
    # update time
    current_time += dt

cv2.destroyAllWindows()

