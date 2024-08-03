import torch
from torchvision import models, transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from image_segmentation import segment_image,  image_resize, create_binary_mask, apply_mask
def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    # success, image = vidcap.read()
    image = image_resize(image, width = 256, height = 256)
    # crop it into 256*256
    # image = image[0:256, 0:256]
    count = 0
    while success:
        image =cv2.resize(image, (256, 256))
        # image = image_resize(image, width = 256, height = 256)

        cv2.imwrite(os.path.join(output_dir, f"frame_{count:05d}.jpg"), image)
        success, image = vidcap.read()
        # image = image_resize(image, width = 256, height = 256)

        count += 1
    vidcap.release()

def frames_segment(image_folder,model, mp4_name):
    #transform the image in the folder
    for i in range(0, len(os.listdir(image_folder))):
        image_path = os.path.join(image_folder, f"frame_{i:05d}.jpg")
        output_predictions = segment_image(image_path, model)
        binary_mask_image = create_binary_mask(output_predictions, class_index = 13)

        masked_image = apply_mask(image_path, binary_mask_image)
        base_masked_name = './VideoTransfer/masked_frames/' + mp4_name
        
        os.makedirs(base_masked_name,exist_ok=True)
        masked_image_folder = os.path.join(base_masked_name, f"frame_{i:05d}.jpg")
        masked_image.save(masked_image_folder)
    
    return binary_mask_image, base_masked_name

def frames_to_video(image_folder, video_name):
    video_name = os.path.abspath(video_name)
    print(video_name)
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    codecs = ['mp4v', 'XVID', 'MJPG']
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))
        
        if video.isOpened():
            print(f"VideoWriter opened successfully with codec {codec}")
            break
    else:
        raise ValueError(f"Could not open VideoWriter with any of the codecs: {codecs}")

    for image in images:
        print(image)
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

    return video_name

frames_to_video('./results/horse2zebra_pretrained/test_latest/images/transformed_frames', './horse2zebra.mp4')