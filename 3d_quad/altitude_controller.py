import numpy as np




class AltitudeController:
    """
    Cascaded altitude controller (position -> velocity -> acceleration -> thrust)
    with tilt compensation using the body z-axis projection onto gravity, similar to PX4.

    Usage: thrust_accel = ctrl.update(z, z_sp, vz, vz_sp=0.0, q=None, dt=None)
    - Returns thrust_accel_sp (T/m) in m/s^2, already tilt-compensated.
    - If mass is set, ctrl.thrust_sp (Newtons) is also provided.
    Conventions: frame 'NED' (Z down, default) or 'ENU' (Z up).
    """
    def __init__(
        self,
        frame: str = 'NED',
        mass: float | None = None,
        gravity: float = 9.81,
        return_normalized: bool = True,
        # Position -> velocity loop
        kp_pos_z: float = 3.0,      # [1/s]
        v_z_max: float = 3.0,       # [m/s]
        # Velocity -> acceleration loop (PI)
        kp_vel_z: float = 6.0,      # [1/s]
        ki_vel_z: float = 2.0,      # [1/s^2]
        az_max: float = 6.0,        # [m/s^2]
        # Thrust limits in accel domain (T/m)
        thrust_accel_min: float = 0.0,
        thrust_accel_max: float = 2.5 * 9.81,
        # Tilt compensation safety
        min_cos_tilt: float = 0.1
    ):
        self.frame = frame.upper()
        assert self.frame in ('NED', 'ENU'), "frame must be 'NED' or 'ENU'"
        self.mass = mass
        self.g = gravity
        self.return_normalized = return_normalized

        # Gains and limits
        self.kp_pos_z = kp_pos_z
        self.v_z_max = abs(v_z_max)
        self.kp_vel_z = kp_vel_z
        self.ki_vel_z = ki_vel_z
        self.az_max = abs(az_max)
        self.thrust_accel_min = thrust_accel_min
        self.thrust_accel_max = thrust_accel_max
        self.min_cos_tilt = min_cos_tilt

        # Internal states/outputs
        self.vz_i = 0.0
        self.vz_sp_out = 0.0
        self.az_sp_out = 0.0
        self.thrust_accel_sp = 0.0  # T/m (m/s^2)
        self.thrust_sp = None       # N if mass is set
        self.thrust_u = 0.0         # normalized [-1,0] in NED (motors up), [0,1] in ENU

    
    def update(self, z, z_sp, vz, vz_sp=0.0, q=None, dt=None, az_sp_ff: float = 0.0):
        # Position -> velocity
        pos_err = z_sp - z
        vz_sp_cmd = vz_sp + self.kp_pos_z * pos_err
        self.vz_sp_out = float(np.clip(vz_sp_cmd, -self.v_z_max, self.v_z_max))

        # Velocity -> acceleration (PI)
        vel_err = self.vz_sp_out - vz
        if dt is not None and dt > 0 and self.ki_vel_z > 0:
            self.vz_i += vel_err * dt
            i_limit = self.az_max / max(self.ki_vel_z, 1e-6)
            self.vz_i = float(np.clip(self.vz_i, -i_limit, i_limit))
        i_term = self.ki_vel_z * self.vz_i
        az_sp = self.kp_vel_z * vel_err + i_term + az_sp_ff
        self.az_sp_out = float(np.clip(az_sp, -self.az_max, self.az_max))

        # Convert to unified down-positive for thrust calculation
        az_down = self.az_sp_out if self.frame == 'NED' else -self.az_sp_out

        # Tilt compensation using body z-axis vs gravity direction
        cos_tilt = self._compute_cos_tilt(q) if q is not None else 1.0
        cos_tilt = float(np.clip(cos_tilt, self.min_cos_tilt, 1.0))

        # T/m required: az_down = g - (T/m) * cos_tilt  =>  T/m = (g - az_down)/cos_tilt
        thrust_accel = (self.g - az_down) / cos_tilt
        self.thrust_accel_sp = float(np.clip(thrust_accel, self.thrust_accel_min, self.thrust_accel_max))

        # If mass known, produce Newtons too
        if self.mass is not None:
            self.thrust_sp = self.mass * self.thrust_accel_sp
        else:
            self.thrust_sp = None

        # Normalized thrust command: map to [-1,0] for NED so that
        #   0   = no thrust, -1 = max upward thrust (motors up, Z down)
        # For ENU, map to [0,1] (upward thrust positive).
        if self.frame == 'NED':
            self.thrust_u = float(np.clip(-self.thrust_accel_sp / max(self.thrust_accel_max, 1e-6), -1.0, 0.0))
        else:  # ENU
            self.thrust_u = float(np.clip(self.thrust_accel_sp / max(self.thrust_accel_max, 1e-6), 0.0, 1.0))

        return self.thrust_u if self.return_normalized else self.thrust_accel_sp
    
    # --------------------------- Helper methods ---------------------------
    @staticmethod
    def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
        """Quaternion [w, x, y, z] -> rotation matrix R (world from body)."""
        q = np.asarray(q, dtype=float).flatten()
        assert q.size == 4, "q must be [w,x,y,z]"
        w, x, y, z = q
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        R = np.array([
            [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
            [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
            [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz]
        ])
        return R

    def _compute_cos_tilt(self, q: np.ndarray) -> float:
        """
        cos(tilt) = projection of body z-axis (FRD down) onto gravity direction.
        - For NED: gravity_dir = +Z (0,0,1), cos_tilt = b_z · gravity_dir.
        - For ENU: gravity_dir = -Z (0,0,-1).
        """
        R = self._quat_to_rotmat(q)
        b_z_world = R[:, 2]  # body z-axis expressed in world
        if self.frame == 'NED':
            gravity_dir = np.array([0.0, 0.0, 1.0])
        else:
            gravity_dir = np.array([0.0, 0.0, -1.0])
        cos_tilt = float(np.dot(b_z_world, gravity_dir))
        return abs(cos_tilt)
       