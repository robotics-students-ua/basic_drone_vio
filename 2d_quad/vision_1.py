import numpy as np

# translation vector of the drone in 2d: [x, z], NED (z down), so 2 m above the ground is z = -2
d = np.array([0.4, -2])
# rotation matrix of the drone in 2d
theta = 0.1
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])


focus_distance = 5.0

# position of ancor points in global ( assume on line z = 0)
anchor_points_global = np.array([[0, 0],
                                  [1, 0],
                                  [2, 0],
                                  [3, 0]])

# lets transform the anchor points to the drone's local frame
anchor_points_local = anchor_points_global - d
anchor_points_local = anchor_points_local @ R.T

# lets find the projection of the anchor points onto the image plane
anchor_points_image = focus_distance *anchor_points_local[:, 0] / anchor_points_local[:, 1]


# --- POSE ESTIMATION FROM IMAGE PROJECTIONS ---
from scipy.optimize import minimize

def project_points(anchor_points_global, d, theta, focus_distance):
    """
    Projects global anchor points to the image plane given pose (d, theta).
    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    anchor_points_local = anchor_points_global - d
    anchor_points_local = anchor_points_local @ R.T
    # Avoid division by zero
    y = anchor_points_local[:, 1]
    x = anchor_points_local[:, 0]
    proj = focus_distance * x / y
    return proj

def pose_error(params, anchor_points_global, observed_projections, focus_distance):
    """
    Error function for pose estimation.
    params: [d_x, d_z, theta]
    """
    d = np.array([params[0], params[1]])
    theta = params[2]
    projected = project_points(anchor_points_global, d, theta, focus_distance)
    # Only compare where y != 0 to avoid inf/nan
    valid = np.isfinite(projected) & np.isfinite(observed_projections)
    return np.sum((projected[valid] - observed_projections[valid])**2)

# Simulate observed projections (in real case, these come from the camera)
observed_projections = anchor_points_image.copy()

# Initial guess for pose: [d_x, d_z, theta]
init_guess = [0.5, -1.5, 0.1]

# Run optimization to estimate pose
res = minimize(
    pose_error,
    init_guess,
    args=(anchor_points_global, observed_projections, focus_distance),
    method='BFGS'
)

estimated_d = res.x[:2]
estimated_theta = res.x[2]

print("True translation:", d)
print("Estimated translation:", estimated_d)
print("True theta:", theta)
print("Estimated theta:", estimated_theta)
print("BFGS iterations:", res.nit)


# --- GAUSS-NEWTON (chapter 4) ---
# u_i = f * tan(alpha_i - theta), alpha_i = atan2(l_i - x, h), h = -z

def projection_jacobian(anchor_points_global, d, theta, focus_distance):
    """
    Jacobian of the projections w.r.t. [d_x, d_z, theta], one row per anchor point.
    """
    h = -d[1]
    delta = anchor_points_global[:, 0] - d[0]
    rho2 = delta**2 + h**2
    u = project_points(anchor_points_global, d, theta, focus_distance)
    k = focus_distance * (1 + (u / focus_distance)**2)
    return np.column_stack([-k * h / rho2, k * delta / rho2, -k])

def gauss_newton(anchor_points_global, observed_projections, focus_distance, init_guess, max_iter=20, tol=1e-10):
    s = np.array(init_guess, dtype=float)
    for i in range(max_iter):
        r = observed_projections - project_points(anchor_points_global, s[:2], s[2], focus_distance)
        J = projection_jacobian(anchor_points_global, s[:2], s[2], focus_distance)
        step = np.linalg.solve(J.T @ J, J.T @ r)
        s += step
        if np.linalg.norm(step) < tol:
            break
    return s, i + 1

gn_pose, gn_iters = gauss_newton(anchor_points_global, observed_projections, focus_distance, init_guess)
print("Gauss-Newton estimated translation:", gn_pose[:2])
print("Gauss-Newton estimated theta:", gn_pose[2])
print("Gauss-Newton iterations:", gn_iters)
