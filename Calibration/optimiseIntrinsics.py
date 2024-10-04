import cv2
from cv2 import aruco
import numpy as np
import sksurgeryimage.calibration.charuco_point_detector as cpd
import sksurgerycalibration.video.video_calibration_driver_mono as mc
import sksurgerycalibration.video.video_calibration_utils as surg_utils
import pathlib
import pandas as pd

# Calibrating using charuco board
chessboard_corners = (11,8)
min_points_to_detect = 20
square_size_mm = (7, 4.2)
dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_50)

detector = cpd.CharucoPointDetector(dictionary, chessboard_corners, square_size_mm)
calibrator = mc.MonoVideoCalibrationDriver(detector, min_points_to_detect)

# Store detected corners and object points
detected_image_points = []
detected_object_points = []

def process_frame(framepath):
    frame = cv2.imread(framepath)
    if frame is None:
        print(f"Frame {framepath} not found")
        return None, None, None

    ids, object_points, image_points = detector.get_points(frame)
    if ids is not None and ids.shape[0] >= min_points_to_detect:
        return frame, image_points, object_points
    return None, None, None

def main(image_folder):
    print(f"Image Folder: {image_folder}")

    # Manually filtered images
    filtered_images = [str(p) for p in pathlib.Path(image_folder).glob("*.png")]

    for framepath in filtered_images:
        frame, image_points, object_points = process_frame(framepath)
        if frame is not None:
            detected_image_points.append(image_points)
            detected_object_points.append(object_points)
            calibrator.grab_data(frame)

    if calibrator.get_number_of_views() >= 2:
        print(f"Total number of views: {calibrator.get_number_of_views()}")
        proj_err, params = calibrator.calibrate()
        print(f"Projection error: {proj_err}")
        print(f'Intrinsics are: \n  {params.camera_matrix}')
        print(f'Distortion matrix is:  \n {params.dist_coeffs}')

        # Save intrinsics
        with open(image_folder / "intrinsics.txt", "a") as intrinsics_file:
            intrinsics_file.write(f"Reprojection error: {proj_err} \n")
            intrinsics_file.write(f'Intrinsics are: \n  {params.camera_matrix}\n')
            intrinsics_file.write(f'Distortion matrix is:  \n {params.dist_coeffs}\n')

        # Save extrinsics
        extr_list = []
        for i in range(len(params.rvecs)):
            extrinsics = surg_utils.extrinsic_vecs_to_matrix(params.rvecs[i], params.tvecs[i])
            extr_list.append([filtered_images[i]] + extrinsics.flatten().tolist())
        columns = ['Frame'] + [f'a{i}{j}' for i in range(1, 5) for j in range(1, 5)]
        df = pd.DataFrame(extr_list, columns=columns)
        df.to_csv(image_folder / 'extrinsics.csv', index=False)
        print(f'Extrinsics saved.')

        # Reproject the object points
        for i, framepath in enumerate(filtered_images):
            frame = cv2.imread(framepath)
            if frame is None:
                print(f"Frame {framepath} not found or unable to read.")
                continue

            image_points, _ = cv2.projectPoints(detected_object_points[i], params.rvecs[i], params.tvecs[i],
                                                params.camera_matrix, params.dist_coeffs)
            image_points = image_points.reshape(-1, 2)

            # Draw detected corners and lines linking to reprojected points
            for j, point in enumerate(detected_image_points[i]):
                cv2.circle(frame, tuple(point.astype(int)), 5, (0, 255, 0), -1)
                cv2.line(frame, tuple(point.astype(int)), tuple(image_points[j].astype(int)), (255, 0, 0), 2)

            # Draw reprojected points
            for point in image_points:
                cv2.circle(frame, tuple(point.astype(int)), 5, (0, 0, 255), -1)

            # Save or display the image
            # output_path = image_folder / f"result_{pathlib.Path(framepath).stem}.png"
            # cv2.imwrite(str(output_path), frame)
            cv2.imshow(f'Result Frame {pathlib.Path(framepath).stem}', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print('Camera Calibration Complete!')
    else:
        print("Not enough views found.")


if __name__ == '__main__':
    main(pathlib.Path("../frames/optframes")) #path to your improved frames
