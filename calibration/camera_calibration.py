import numpy as np
import cv2
import glob


def main():
    # --- User-defined parameters ---
    # Number of internal corners in the chessboard
    # For a 9x6 board, you have 9 internal corners along width, 6 along height
    # Which means, 10x7 boxes (we don't include the boxes at the border)
    chessboard_size = (9, 6)

    # Size of each square in your checkerboard (in some consistent unit, e.g. millimeters).
    # If you only care about fx, fy, cx, cy (no real-world scaling), you can keep it at 1.0.
    square_size = 0.26 # in meters

    # Path to your calibration images
    # e.g., "calibration_images/*.jpg" if you have multiple .jpg files in that folder
    images_path_pattern = "calibration_images/*.png"

    # --- Prepare 3D points of the corners in real-world space ---
    # For a board of (9x6), we have 9 corners along one axis, 6 along the other.
    # We'll create a list of points like (0,0,0), (1,0,0), (2,0,0) ... (8,5,0) multiplied by square_size
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store 3D points and 2D points for all images
    obj_points = []  # 3D points in real-world
    img_points = []  # 2D points in image plane

    # Get all image file paths
    images = glob.glob(images_path_pattern)

    # Criteria for corner sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for file_name in images:
        img = cv2.imread(file_name)
        if img is None:
            print(f"Failed to load {file_name}")
            continue
        print("reading image", file_name)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Refine corner locations to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners_refined)

            # Optionally, draw and display corners for verification
            cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
            cv2.imshow(file_name, img)
            cv2.waitKey(1000)
        else:
            print(f"Chessboard not found in {file_name}")

    cv2.destroyAllWindows()

    if len(obj_points) < 1:
        print("No chessboard corners found. Calibration aborted.")
        return

    # --- Run camera calibration ---
    # ret: RMS reprojection error
    # camera_matrix: 3x3 internal camera matrix
    # dist_coeffs: distortion coefficients
    # rvecs, tvecs: rotation and translation vectors for each image
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    print("Calibration successful!")
    print(f"RMS Reprojection Error: {ret}")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients (k1, k2, p1, p2, k3, ...):")
    print(dist_coeffs.ravel())

    # Extract fx, fy, cx, cy from camera_matrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print(f"\nIntrinsic parameters:")
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")
    print("shapes", gray.shape[::-1])


if __name__ == "__main__":
    main()
