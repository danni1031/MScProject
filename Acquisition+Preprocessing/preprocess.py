import os
import random
from PIL import Image
import shutil
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, GaussianBlur
import torchvision.utils
from torchvision.transforms import ToPILImage
from PIL import Image, UnidentifiedImageError
from torchvision.utils import make_grid, save_image
import cv2
import numpy as np
import pathlib

def mask(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.zeros_like(img_rgb)
    height, width = img_rgb.shape[:2]
    center = (width // 2, height // 2)
    radius = min(center[0]-30, center[1]-30, width - center[0]-30, height - center[1]-30)

    # Create circular mask
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # Manually exclude the blue corner
    mask[0:int(0.056*height), 0:int(0.5*width)] = 0

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_rgb, mask)

    return masked_img

def convert_to_greyscale_and_crop(frame):
    # Convert to greyscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the original dimensions
    h, w = gray_frame.shape[:2]

    # Determine the size of the square (shortest side)
    size = min(h, w)

    # Calculate the coordinates to center-crop the image
    top = (h - size) // 2
    left = (w - size) // 2
    bottom = top + size
    right = left + size

    # Perform the crop
    cropped_frame = gray_frame[top:bottom, left:right]

    return cropped_frame

def video2image(viddirectory, imgdirectory):
    video = cv2.VideoCapture(viddirectory.__str__())
    count = 0
    while video.isOpened() and count < 700:
        ret, frame = video.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        masked_frame = mask(frame)
        final_frame = convert_to_greyscale_and_crop(masked_frame)
        
        img_path = imgdirectory / f'{count}.png'
        cv2.imwrite(img_path.__str__(), final_frame)

        count += 1
        
    video.release()
    print("Video released")

def process_videos(input_folders_file, output_folders_file, parent_output_dir):
    with open(input_folders_file, 'r') as infile, open(output_folders_file, 'r') as outfile:
        input_dirs = infile.read().splitlines()
        output_dirs = outfile.read().splitlines()
        
        if len(input_dirs) != len(output_dirs):
            raise ValueError("Input and output directories count do not match")
        
        for input_dir, output_dir in zip(input_dirs, output_dirs):
            print(f"Processing {input_dir} to {output_dir}")
            input_dir_path = pathlib.Path(input_dir) / 'video'
            output_dir_path = pathlib.Path(parent_output_dir) / output_dir  # Add parent directory here
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Assuming video files are inside the input directory
            for video_file in input_dir_path.glob('*.avi'):
                video2image(video_file, output_dir_path)
            print(f"Processed {input_dir} to {output_dir}")

if __name__ == '__main__':
    input_folders_file = 'D:/PitSim/scripts/vid1.txt'  # Path to the text file with input folders
    output_folders_file = 'D:/PitSim/scripts/frame1.txt'  # Path to the text file with output folders
    parent_output_dir = 'D:/PitSim/frames'  # Parent directory for all output directories
    process_videos(input_folders_file, output_folders_file, parent_output_dir)
    print('All videos processed successfully.')