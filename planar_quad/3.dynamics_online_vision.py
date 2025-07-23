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




# visulize the ground

world_size = 200
np.random.seed(42)  # for reproducibility
# generate binary string for ground
world = np.random.randint(0, 2, world_size)
square_size = 2.2  # size of each square in meters
features_locations = np.where(np.diff(world) != 0)[0] * square_size  # locations of features in meters
feature_locations = features_locations + x_min # shift to match x_min
features = np.diff(world)
features = features[features != 0]

print(world)

# camera parameters
camera_width = 800
camera_height = 600
camera_fov = 60  # field of view in degrees
camera_focal_length = camera_width / (2 * np.tan(np.radians(camera_fov / 2)))

def project_features_to_camera(drone_x, drone_z, drone_theta, features_x):
    """Project feature points from global frame to camera image coordinates"""
    camera_img = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
    
    # Camera is at drone position, facing downward with rotation -theta
    projected_features = []
    
    for feat_x in features_x:
        # Feature is at (feat_x, 0) in global frame (on ground, z=0)
        # Transform to drone-centered coordinates
        dx = feat_x - drone_x
        dz = 0 - drone_z  # feature at ground level (z=0), drone at drone_z
        
        # Rotate by -drone_theta (camera rotation opposite to drone rotation)
        cos_theta = np.cos(-drone_theta)
        sin_theta = np.sin(-drone_theta)
        
        # Rotate in horizontal plane (x-z plane)
        cam_x = dx * cos_theta - 0 * sin_theta  # dz_horizontal = 0 since feature is at same ground level
        cam_z = dx * sin_theta + 0 * cos_theta
        cam_y = dz  # vertical distance (altitude)
        
        # Only project features that are in front of camera and drone is above ground
        if cam_y > 0 and drone_z < 0:  # camera looking down, drone above ground
            # Perspective projection
            # Project to normalized camera coordinates
            x_norm = cam_x / cam_y
            z_norm = cam_z / cam_y
            
            # Convert to pixel coordinates
            u = camera_focal_length * x_norm + camera_width / 2
            v = camera_focal_length * z_norm + camera_height / 2
            
            # Check if feature is within camera view
            if 0 <= u < camera_width and 0 <= v < camera_height:
                projected_features.append((int(u), int(v)))
    
    # Draw features on camera image
    for i, (u, v) in enumerate(projected_features):
        # Different colors for different feature types
        if i < len(features) and features[i] == 1:  # rising edge (black to white)
            feature_color = (0, 255, 0)  # green
        elif i < len(features) and features[i] == -1:  # falling edge (white to black)
            feature_color = (0, 0, 255)  # red
        else:
            feature_color = (255, 255, 0)  # yellow for other features
            
        # Draw feature with color coding
        cv2.circle(camera_img, (u, v), 8, feature_color, -1)
        cv2.circle(camera_img, (u, v), 12, (255, 255, 255), 2)  # white border
        
        # Add feature index label
        cv2.putText(camera_img, str(i), (u + 15, v - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add feature legend
    cv2.putText(camera_img, "Green: B->W edge", (10, camera_height - 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(camera_img, "Red: W->B edge", (10, camera_height - 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(camera_img, "Yellow: Other", (10, camera_height - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Add crosshair at center
    cv2.line(camera_img, (camera_width//2 - 20, camera_height//2), 
             (camera_width//2 + 20, camera_height//2), (255, 0, 0), 2)
    cv2.line(camera_img, (camera_width//2, camera_height//2 - 20), 
             (camera_width//2, camera_height//2 + 20), (255, 0, 0), 2)
    
    return camera_img, projected_features



# create OpenCV windows
cv2.namedWindow('Drone Simulation', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Camera View', cv2.WINDOW_AUTOSIZE)

# simulation loop
while current_time < max_sim_time:
    # create blank image with gray background
    img = np.full((window_height, window_width, 3), (64, 64, 64), dtype=np.uint8)  # dark gray background
    
    # draw ground line (z = 0)
    ground_y = center_y - int(0 * scale)
    cv2.line(img, (0, ground_y), (window_width, ground_y), (100, 100, 100), 2)
    
    # draw coordinate system
    cv2.line(img, (center_x, 0), (center_x, window_height), (50, 50, 50), 1)
    cv2.line(img, (0, center_y), (window_width, center_y), (50, 50, 50), 1)
    
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
    

    T_z = m * g + thrust_offset  # total thrust with user control
    M_y = torque_input  # torque with user control

    # T_z = -(u1 + u2)  # total thrust/ minus since motors upwards, z downwards
    # M_y = r * (u1 - u2)  # torque, u1 is forward motor, u2 is backward motor

    # control allocation, solve linear system for u1 and u2
    u1, u2 = np.linalg.solve(np.array([[-1, -1], [r, -r]]), np.array([-T_z, M_y]))

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
    
    # draw ground pattern visualization as black and white squares
    # size_ww = 100
    # pattern_start_x = max(x_min, int((x - size_ww) / square_size))
    # pattern_end_x = min(world_size - 1, int((x + size_ww) / square_size))
    
    # square_height = 8  # height of squares in pixels
    
    # for i in range(pattern_start_x, pattern_end_x):
    #     pattern_x = i * square_size
    #     pattern_screen_x = int(center_x + pattern_x * scale)
    #     pattern_screen_x_end = int(center_x + (pattern_x + square_size) * scale)
        
    #     if 0 <= pattern_screen_x < window_width:
    #         if world[i] == 1:  # white square
    #             cv2.rectangle(img, (pattern_screen_x, ground_y - square_height//2), 
    #                         (pattern_screen_x_end, ground_y + square_height//2), 
    #                         (255, 255, 255), -1)  # white
    #         else:  # black square
    #             cv2.rectangle(img, (pattern_screen_x, ground_y - square_height//2), 
    #                         (pattern_screen_x_end, ground_y + square_height//2), 
    #                         (0, 0, 0), -1)  # black
            
    #         # draw square borders for clarity
    #         cv2.rectangle(img, (pattern_screen_x, ground_y - square_height//2), 
    #                     (pattern_screen_x_end, ground_y + square_height//2), 
    #                     (128, 128, 128), 1)  # gray border
    
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
        f"Position: ({x:.2f}, {z:.2f})",
        f"Velocity: ({x_dot:.2f}, {z_dot:.2f})",
        f"Angle: {theta:.3f} rad",
        f"Angular speed: {theta_dot:.3f} rad/s",
        f"Controls: u1={u1:.2f}, u2={u2:.2f}",
        f"Thrust offset: {thrust_offset:.2f}",
        f"Torque input: {torque_input:.3f}",
        f"Total features: {len(features_locations)}",
        "",
        "Feature Colors:",
        "Green: Black->White edge",
        "Red: White->Black edge", 
        "",
        "Controls:",
        "UP/DOWN: Thrust +/-",
        "LEFT/RIGHT: Torque +/-",
        "R: Reset simulation",
        "ESC: Exit"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(img, text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # generate camera view with projected features
    camera_view, projected_features = project_features_to_camera(x, z, theta, features_locations)
    
    # add camera info to camera view
    cv2.putText(camera_view, f"Altitude: {-z:.1f}m", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(camera_view, f"Heading: {np.degrees(theta):.1f}°", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(camera_view, f"Features: {len(projected_features)}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # show images
    cv2.imshow('Drone Simulation', img)
    cv2.imshow('Camera View', camera_view)
    
    # handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == 82 or key == ord('w'):  # UP arrow or W key
        thrust_offset += thrust_step
    elif key == 84 or key == ord('s'):  # DOWN arrow or S key
        thrust_offset -= thrust_step
    elif key == 81 or key == ord('a'):  # LEFT arrow or A key
        torque_input += torque_step
    elif key == 83 or key == ord('d'):  # RIGHT arrow or D key
        torque_input -= torque_step
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

