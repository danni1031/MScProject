import cv2 as cv
from sksurgerynditracker.nditracker import NDITracker
# from sksurgerytrackervisualisation.ui import sksurgerytrackervisualisation_cl
import keyboard
import pathlib
import sys
import serial
import numpy as np
import pandas as pd
import time
import datetime

def main():
    if pathlib.Path("D:\PitSim\data").exists():
        out_path = pathlib.Path("D:\PitSim\data")
    else:
        out_path = pathlib.Path("./acquisition")

    #acquisition_path = out_path / "QS"

    date_path = out_path / datetime.date.today().__str__()
    time_path = date_path / datetime.datetime.now().strftime("%H%M").__str__()

    video_path = time_path / "video"
    em_path = time_path / "EM"
    force_path = time_path / "force"

    if not out_path.exists():
        out_path.mkdir()
    
    if not date_path.exists():
        date_path.mkdir()

    if not time_path.exists():
        time_path.mkdir()

    if not video_path.exists():
        video_path.mkdir()

    if not em_path.exists():
        em_path.mkdir()
        
    if not force_path.exists():
        force_path.mkdir()

    # ---- Set force object ----
    # Configure the serial port
    serial_port = 'COM4'  # Replace with your Arduino's serial port
    baud_rate = 115200  # Must match the baud rate set on the Arduino
    timeout = 1  # Timeout value for serial communication
    
    # ----- Open VideoCapture object -----

    # Select camera ID
    video_capture = cv.VideoCapture(0, cv.CAP_DSHOW)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH,1920)
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT,1080) 

    # We need to check if camera is already open
    if (video_capture.isOpened() == False): 
        print("Error reading video file") 

    # Set video resolutions
    frame_width = int(video_capture.get(3)) #1920
    frame_height = int(video_capture.get(4)) #1080
    size = (1920, 1080) 

    result = cv.VideoWriter(str(video_path/'video.avi'), 
                            cv.VideoWriter_fourcc(*'X264'), 
                            20, size) 

    # ----- Open NDITracker object -----

    settings = {
        "tracker type": "aurora",
        "ports to probe": 2,
        "verbose": True,
        "use quaternions": False,
        "smoothing buffer": 2
    }
    tracker = NDITracker(settings)
    tracker.start_tracking()


    # ----- Start capture and trackingq -----
    recording_count = 0
    count = 0
    data_list = []
    times_list = []
    force_list = []
    current_time_start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


    ser = serial.Serial(serial_port, baud_rate, timeout=timeout)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    force_list = []
    start, end = 0, 0
    while True:
        if count == 0:
            starttime = time.time()
        now = time.time()-starttime
        data = ser.readline().decode().strip()
        ser.flush()
        force_list.append([now] + [data])
        
        ret, frame = video_capture.read()

        if not ret:
            print("In capture_frame(): cv.VideoCapture.read() failed.")
            return cv.Mat

        port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
        end = timestamps[0]
        if count == 0:
            time_begin = time.time()

        for t in tracking:
            print(t)

        if ret is True:

            # Write the frame into the file 
            result.write(frame) 
        
        for track in tracking:
            # Extract a11 to a33 and x, y, z for each tracker
            a_values = track[:3, :3].flatten()
            xyz_values = track[:3, 3]
            
            # Combine a_values and xyz_values into one list
            combined_values = np.concatenate((a_values, xyz_values))
            
            # Append the timestamp (or count) and combined values to the data list
            data_list.append([timestamps[0]-starttime]+[count] + combined_values.tolist())

        count += 1
        fps = 1/(end-start)
        start = timestamps[0]
        times_list.append([fps])


        if keyboard.is_pressed('q'):
            break
    
    #Write force dataframe into csv format
    df_force = pd.DataFrame(force_list)
    df_force.to_csv(force_path / 'force.csv',index=False)
    
    #Write EM dataframe into csv format
    columns = ['Timestamp'] + ['Count'] + [f'a{i}{j}' for i in range(1, 4) for j in range(1, 4)] + ['x', 'y', 'z']
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(em_path / 'em_data.csv', index=False)
    df_times = pd.DataFrame(times_list,columns = ['times'])
    df_times.to_csv(em_path/'fps.csv', index=False)
    
    #Needed for force team?
    current_time_end = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    file_path = f'temporalb_{recording_count}.txt'
    with open(force_path / file_path, 'w') as file:
        file.write(str(recording_count))
        file.write(current_time_start)
        file.write(current_time_end)
        
    # ----- Close serial object -----
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.close()
    
    
    # ----- Close VideoCapture object -----

    video_capture.release()
    cv.destroyAllWindows()
    result.release() 

    # ----- Close NDITracker object -----

    tracker.stop_tracking()
    tracker.close()

    sys.exit(0)


if __name__ == '__main__':
    main()

