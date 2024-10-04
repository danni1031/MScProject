import cv2
from cv2 import aruco
import numpy as np
import sksurgeryimage.calibration.charuco_point_detector as cpd
import sksurgerycalibration.video.video_calibration_driver_mono as mc
import sksurgerycalibration.video.video_calibration_utils as surg_utils
import pathlib
from tqdm import tqdm
import torch
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd

# Calibrating using charuco board
chessboard_corners = (11, 8)
min_points_to_detect = 35
square_size_mm = (7, 4.2)  
dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_50)

detector = cpd.CharucoPointDetector(dictionary, chessboard_corners, square_size_mm)
calibrator = mc.MonoVideoCalibrationDriver(detector, min_points_to_detect)

# Store detected corners and object points
detected_image_points = []
detected_object_points = []

def process_frame(frame_count, video_source):
    framepath = str(video_source / f"{frame_count+19}.png")
    frame = cv2.imread(framepath)

    if frame is None:
        print(f"Frame {frame_count} not found")
        return None
    
    number_of_points = calibrator.grab_data(frame)
    
    if number_of_points > 0:
        return frame_count

def main(video_source):
    print(f"Video Source: {video_source}")
    start_time = time.time()  # Start time measurement

    num_frames = 561  # total number of frames
    num_processes = cpu_count()  # Use all available CPU cores
    part = partial(process_frame, video_source=video_source)
    
    print("Beginning pooling")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(part, range(num_frames)), total=num_frames))
    print("Finished processing pool")
    
    result_list = [frame_count for frame_count in results if frame_count is not None]
    print(result_list)
    
    for frame_count in result_list:
        if frame_count is None:
            continue
        framepath = str(video_source / f"{frame_count+19}.png")
        frame = cv2.imread(framepath)
        
        if frame is None:
            print(f"Frame {frame_count} not found or unable to read.")
            continue
        
        ids, object_points, image_points = detector.get_points(frame)
        if ids is not None and ids.shape[0] >= min_points_to_detect:
            detected_image_points.append(image_points)
            detected_object_points.append(object_points)
        
        number_of_points = calibrator.grab_data(frame)

    if calibrator.get_number_of_views() >= 2:
        print(f"Total number of views: {calibrator.get_number_of_views()}")
        proj_err, params = calibrator.calibrate()
        print(f"Projection error: {proj_err}")
        print(f'Intrinsics are: \n  {params.camera_matrix}')
        print(f'Distortion matrix is:  \n {params.dist_coeffs}')
        
        # Save intrinsics
        with open(video_source / "intrinsics.txt", "a") as intrinsics_file:
            intrinsics_file.write(f"Reprojection error: {proj_err} \n")
            intrinsics_file.write(f'Intrinsics are: \n  {params.camera_matrix}\n')
            intrinsics_file.write(f'Distortion matrix is:  \n {params.dist_coeffs}\n')

        # Save extrinsics
        extr_list = []
        for i in range(len(params.rvecs)):
            extrinsics = surg_utils.extrinsic_vecs_to_matrix(params.rvecs[i], params.tvecs[i])
            extr_list.append([result_list[i]] + extrinsics.flatten().tolist())
        columns = ['Frame'] + [f'a{i}{j}' for i in range(1, 5) for j in range(1, 5)]
        df = pd.DataFrame(extr_list, columns=columns)
        df.to_csv(video_source / 'extrinsics.csv', index=False)
        print(f'Extrinsics saved.')

        # Reproject the object points
        for i, frame_count in enumerate(result_list):
            framepath = str(video_source / f"{frame_count+19}.png")
            frame = cv2.imread(framepath)
            
            if frame is None:
                print(f"Frame {frame_count+19} not found or unable to read.")
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
            output_path = video_source / f"result_00{frame_count}.png"
            cv2.imwrite(str(output_path), frame)
            cv2.imshow(f'Result Frame {frame_count}', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print('Camera Calibration Complete!')
    else:
        print("Not enough views found.")
    
    end_time = time.time()  # End time measurement
    print(f"Total computational time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main(pathlib.Path("../frames/trial"))
