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
z_min = -100  # minimum z position 
z_max = 0.0  # maximum z position (ground level)
x_min = -100  # minimum x position
x_max = 100  # maximum x position

# visualization parameters
window_width = 1200
window_height = 600
scale = 5  # pixels per meter
center_x = window_width // 2
center_y = int(window_height * 0.9)  # center of the window in y-axis

# for storing trajectory
trajectory = []
current_time = 0.0

# control inputs (can be modified by keyboard)
thrust_offset = 0.0 # additional thrust beyond gravity compensation
torque_input = 0.0 # torque input
thrust_step = 0.1  # how much thrust changes per key press
torque_step = 0.001  # how much torque changes per key press




# visulize the ground
world_size = 200
np.random.seed(42)  # for reproducibility
# generate binary string for ground
# world = np.random.randint(0, 2, world_size)
# world as alternating black and white squares
world = np.array([1, 0] * (world_size // 2))[:world_size]  # alternating pattern

square_size = 1.2  # size of each square in meters
features_locations = np.where(np.diff(world) != 0)[0] * square_size  # locations of features in meters
feature_locations = features_locations + x_min # shift to match x_min
# add a zero z coordinate 
feature_locations = np.array([[f,0] for f in feature_locations]).T  # convert to 2D array for features
features = np.diff(world)
features = features[features != 0]

focus_distance = 400.0  # distance from camera to focus point
camera_fov = 60  # field of view in degrees

def get_projection(x, z, theta):
    # camera rotation matrix
    R_cam = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])  # rotation matrix for camera frame
    # camera translation vector
    t_cam = np.array([x, z])  # camera position in world coordinates
    phi = np.radians(camera_fov / 2)  # half field of view in radians
    x_min_visible = x + z * np.tan(phi - theta)
    x_max_visible = x - z * np.tan(phi + theta)
    # filter feature_locations within visible range
    visible_features_locations = feature_locations[:, (feature_locations[0] >= x_min_visible) & (feature_locations[0] <= x_max_visible)]
    # filter features that are visible in the camera frame
    visible_features = features[(feature_locations[0] >= x_min_visible) & (feature_locations[0] <= x_max_visible)]

    f_loc_camera_frame = (R_cam @ (visible_features_locations - t_cam[:2, np.newaxis])).T
    projected_features = focus_distance * f_loc_camera_frame[:, 0] / f_loc_camera_frame[:, 1]
    return projected_features, visible_features

def draw_vector(locations, edges):
    image_height = 300
    image_width = 2 * int(np.tan(np.deg2rad(60) / 2) * focus_distance)
    
    if len(locations) == 0:
        return np.zeros((image_height, image_width), dtype=np.uint8)
    
    edges = np.array(edges)  # -1: black→white, 1: white→black
    center = image_width // 2
    image = np.zeros((image_height, image_width), dtype=np.uint8)  # initialize with zeros (black)
    put_pixels = locations + center  # shift to center
    put_pixels = put_pixels.astype(int)  # convert to integer pixel coordinates
    
    current_location = 0
    max_location = put_pixels.max()
    for i in range(window_width):
        if i > max_location:
            if edges[-1] == -1:  # black to white edge
                image[:, i:] = 0  # set pixel to white
            elif edges[-1] == 1:  # white to black edge
                image[:, i:] = 255
            break
        if i < put_pixels[current_location]:
            if edges[current_location] == -1:  # black to white edge
                image[:, i] = 255  # set pixel to white
            elif edges[current_location] == 1:  # white to black edge
                image[:, i] = 0
        else:
            current_location += 1

    for i in range(len(put_pixels) - 1):
        try:
            image[:, put_pixels[i]] = 255  # set pixel to white
        except IndexError:
            pass

    return image

# create OpenCV windows
cv2.namedWindow('Drone Simulation', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Camera View', cv2.WINDOW_AUTOSIZE)

# simulation loop
while current_time < max_sim_time:
    # create blank image with gray background
    img = np.full((window_height, window_width, 3), (64, 64, 64), dtype=np.uint8)  # dark gray background
    
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
    

    T_z = m * g + thrust_offset  # total thrust with user control
    M_y = torque_input  # torque with user control

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
    
    locations, edges = get_projection(x, z, theta)
    camera_view = draw_vector(locations, edges)  # Call the function to visualize the vector

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

