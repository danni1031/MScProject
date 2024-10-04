import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to detect Charuco corners
def detect_charuco_corners(image, charuco_board, aruco_dict):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
        return charuco_corners, charuco_ids
    return None, None

# Function to visualize Charuco corner detection
def visualize_charuco_detection(image, charuco_corners, charuco_ids):
    img = cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
    cv2.imshow('Charuco Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to visualize reprojection
def visualize_reprojection(image, objpoints, imgpoints, ids, camera_matrix, dist_coeffs, rvecs, tvecs):
    imgp2, _ = cv2.projectPoints(objpoints, rvecs, tvecs, camera_matrix, dist_coeffs)

    for p1, p2, id_val in zip(imgpoints, imgp2, ids):
        p1 = tuple(int(coord) for coord in p1.ravel())  # Convert to tuple of integers
        p2 = tuple(int (coord) for coord in p2.ravel())  # Convert to tuple of integers
        cv2.circle(image, p1, 5, (0, 255, 0), -1)  # Original points in green
        cv2.circle(image, p2, 3, (0, 0, 255), -1)  # Reprojected points in red
        cv2.putText(image, str(id_val[0]), (p2[0] + 5, p2[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(image, p1, p2, (0, 165, 255), 1)  # Draw a line between original and reprojected points

    cv2.imshow('Reprojection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to extract transformations
def extract_transformations(df, is_em_data=False):
    rotations = []
    translations = []
    for index, row in df.iterrows():
        if is_em_data:
            # Rotation part
            R = np.array([[row['a11'], row['a12'], row['a13']],
                          [row['a21'], row['a22'], row['a23']],
                          [row['a31'], row['a32'], row['a33']]])
            # Translation part
            t = np.array([row['x'], row['y'], row['z']]).reshape((3, 1))
        else:
            # Rotation part
            R = np.array([[row['a11'], row['a12'], row['a13']],
                          [row['a21'], row['a22'], row['a23']],
                          [row['a31'], row['a32'], row['a33']]])
            # Translation part
            t = np.array([row['a14'], row['a24'], row['a34']]).reshape((3, 1))
        
        rotations.append(R)
        translations.append(t)
    return rotations, translations

# Function to analyze transformations
def analyze_transformations(rotations, translations, label):
    rotation_angles = []
    translation_magnitudes = []
    
    for R in rotations:
        angle = np.arccos((np.trace(R) - 1) / 2)  # Rotation angle in radians
        rotation_angles.append(np.degrees(angle))  # Convert to degrees
    
    for t in translations:
        magnitude = np.linalg.norm(t)
        translation_magnitudes.append(magnitude)
    
    print(f"Analysis of {label}:")
    print(f"Rotation Angles (degrees): min={min(rotation_angles)}, max={max(rotation_angles)}, mean={np.mean(rotation_angles)}")
    print(f"Translation Magnitudes: min={min(translation_magnitudes)}, max={max(translation_magnitudes)}, mean={np.mean(translation_magnitudes)}\n")

def perform_hand_eye_calibration(extrinsics_path, em_data_path):
    extrinsics_df = pd.read_csv(extrinsics_path)
    em_data_df = pd.read_csv(em_data_path)

    # Clean the em_data_df by removing rows with NaN values
    em_data_df_clean = em_data_df.dropna()

    # Extract rotations and translations
    rotations_A, translations_A = extract_transformations(extrinsics_df)
    rotations_B, translations_B = extract_transformations(em_data_df_clean, is_em_data=True)

    # Check if we need to subsample the em_data
    if len(rotations_A) < len(rotations_B):
        em_data_df_sampled = em_data_df_clean.sample(n=len(rotations_A), random_state=42)
        rotations_B_sampled, translations_B_sampled = extract_transformations(em_data_df_sampled, is_em_data=True)
    else:
        rotations_B_sampled, translations_B_sampled = rotations_B, translations_B

    # Analyze transformations
    analyze_transformations(rotations_A, translations_A, "Extrinsics Data")
    analyze_transformations(rotations_B_sampled, translations_B_sampled, "EM Data")

    # Convert to the format required for cv2.calibrateHandEye
    R_gripper2base = np.array(rotations_A)
    t_gripper2base = np.array(translations_A)
    R_target2cam = np.array(rotations_B_sampled)
    t_target2cam = np.array(translations_B_sampled)

    # Testing with different hand-eye calibration methods
    methods = {
        "Tsai": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS
    }

    best_method = None
    best_T_cam2gripper = None
    min_error = float('inf')
    residuals_dict = {}

    for method_name, method in methods.items():
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method
            )
            print(f"Method: {method_name}")
            print("Rotation matrix from camera to gripper:")
            print(R_cam2gripper)
            print("Translation vector from camera to gripper:")
            print(t_cam2gripper)
            print("\n")
            
            # Store the transformation matrix for the selected method
            T_cam2gripper = np.eye(4)
            T_cam2gripper[:3, :3] = R_cam2gripper
            T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

            # Calculate residual errors for this method
            method_residuals = []
            reprojection_error = 0
            for R_b, t_b, R_a, t_a in zip(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
                t_estimated = np.dot(R_cam2gripper, t_b) + t_cam2gripper
                error = np.linalg.norm(t_estimated - t_a)
                reprojection_error += error
                method_residuals.append(error)

            reprojection_error /= len(R_gripper2base)
            residuals_dict[method_name] = method_residuals
            
            if reprojection_error < min_error:
                min_error = reprojection_error
                best_T_cam2gripper = T_cam2gripper
                best_method = method_name

        except cv2.error as e:
            print(f"Method: {method_name} failed with error: {e}")

    print(f"Best hand-eye calibration method: {best_method} with reprojection error: {min_error}")
    return best_T_cam2gripper, best_method, residuals_dict

# Function to draw a coordinate frame
def draw_frame(ax, T, label, length=0.1):
    origin = T[:3, 3]
    x_axis = origin + T[:3, 0] * length
    y_axis = origin + T[:3, 1] * length
    z_axis = origin + T[:3, 2] * length
    
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0] - origin[0], x_axis[1] - origin[1], x_axis[2] - origin[2], color='r')
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0] - origin[0], y_axis[1] - origin[1], y_axis[2] - origin[2], color='g')
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0] - origin[0], z_axis[1] - origin[1], z_axis[2] - origin[2], color='b')
    ax.text(origin[0], origin[1], origin[2], label)

# Function to visualize the camera position
def visualize_camera_position(T_base_to_ee, T_camera_to_ee):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the base frame
    draw_frame(ax, np.eye(4), 'Base')
    
    # Draw the end-effector frame
    draw_frame(ax, T_base_to_ee, 'EE')
    
    # Compute and draw the camera frame in the base coordinate system
    T_base_to_camera = np.dot(T_base_to_ee, T_camera_to_ee)
    draw_frame(ax, T_base_to_camera, 'Camera')
    
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def visualize_residuals(residuals_dict):
    plt.figure()
    for method_name, residuals in residuals_dict.items():
        plt.plot(residuals, marker='o', label=method_name)
    plt.title("Residual Errors for Hand-Eye Calibration")
    plt.xlabel("Image Pair Index")
    plt.ylabel("Residual Error (mm)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Paths to CSV files and images
    extrinsics_path = '../frames/optframes/extrinsics.csv'  
    em_data_path = 'relative_poses_filtered.csv'  
    # Test quality of results with x images
    image_paths = ['../frames/optframes/26.png', '../frames/optframes/33.png', '../frames/optframes/41.png', '../frames/optframes/52.png', '../frames/optframes/80.png']   

    # Define Charuco board parameters based on your previous setup
    chessboard_corners = (11, 8)
    square_size = 0.007  # 7 mm in meters
    marker_size = 0.0042  # 4 mm in meters
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    charuco_board = cv2.aruco.CharucoBoard_create(squaresX=chessboard_corners[0], squaresY=chessboard_corners[1], squareLength=square_size, markerLength=marker_size, dictionary=dictionary)

    # Load images
    images = [cv2.imread(img_path) for img_path in image_paths]

    # Detect Charuco corners
    all_corners = []
    all_ids = []
    for img in images:
        charuco_corners, charuco_ids = detect_charuco_corners(img, charuco_board, dictionary)
        if charuco_corners is not None:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            visualize_charuco_detection(img, charuco_corners, charuco_ids)

    # Check if enough corners are detected
    if len(all_corners) < 2:
        print("Not enough Charuco corners detected in the images.")
        exit()

    # Perform hand-eye calibration and get the transformation matrix
    T_camera_to_ee, best_method, residuals_dict = perform_hand_eye_calibration(extrinsics_path, em_data_path)
    
    if T_camera_to_ee is None:
        print("Hand-eye calibration failed.")
        exit()

    # Perform camera calibration using Charuco corners
    image_size = images[0].shape[1::-1]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners, charucoIds=all_ids, board=charuco_board, imageSize=image_size, cameraMatrix=None, distCoeffs=None)
    
    if not ret:
        print("Camera calibration failed.")
        exit()

    # Visualize reprojection
    for img, corners, ids, rvec, tvec in zip(images, all_corners, all_ids, rvecs, tvecs):
        visualize_reprojection(img, charuco_board.chessboardCorners, corners, ids, camera_matrix, dist_coeffs, rvec, tvec)
    
    # Visualize residuals for all methods
    visualize_residuals(residuals_dict)
    
    print(f"Best hand-eye calibration method: {best_method}")
    print(f"Residuals: {residuals_dict[best_method]}")
