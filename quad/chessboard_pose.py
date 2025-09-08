import cv2
import numpy as np

class ChessboardPoseEstimator:

    def __init__(self, chessboard_size=(9, 6), square_size=1.0, fov_deg=60):
        """
        chessboard_size: (corners_x, corners_y) - number of inner corners per chessboard row and column
        square_size: size of a square in user-defined units (e.g., meters)
        fov_deg: horizontal field of view of the camera in degrees
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.fov_deg = fov_deg
        # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0) * square_size
        objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp = objp * square_size


    def calculate_camera_matrix(self, image_size):
        h, w = image_size
        # Calculate camera matrix from FOV and image size
        fov_rad = np.deg2rad(self.fov_deg)
        fx = w / (2 * np.tan(fov_rad / 2))
        fy = fx
        cx = w / 2
        cy = h / 2
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])
        return camera_matrix

    def estimate_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        if not ret:
            return None, None  

        h, w = gray.shape
        camera_matrix = self.calculate_camera_matrix((h, w))
        dist_coeffs = np.zeros(5)  

        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # Solve PnP
        retval, rvec, tvec = cv2.solvePnP(self.objp, corners2, camera_matrix, dist_coeffs)
        if not retval:
            return None, None
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        return tvec.flatten(), R

    def draw_axes(self, frame, tvec, R_mat, axis_length=3.0):
        """
        Draws the chessboard coordinate axes on the frame.
        X: red, Y: green, Z: blue
        axis_length: length of the axes in chessboard units (default: 3 squares)
        """

        rvec, _ = cv2.Rodrigues(R_mat)

        h, w = frame.shape[:2]
        camera_matrix = self.calculate_camera_matrix((h, w))
        dist_coeffs = np.zeros(5)
        axis = np.float32([
            [0, 0, 0],
            [axis_length * self.square_size, 0, 0],
            [0, axis_length * self.square_size, 0],
            [0, 0, -axis_length * self.square_size]
        ])
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        frame = cv2.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0,0,255), 3, tipLength=0.2)  # X - red
        frame = cv2.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0,255,0), 3, tipLength=0.2)  # Y - green
        frame = cv2.arrowedLine(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255,0,0), 3, tipLength=0.2)  # Z - blue
        return frame