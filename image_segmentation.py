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
# Load a pre-trained segmentation model

# Define the transformation
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def segment_image(image_path, model ):
    # Open the image
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Move the input to the GPU for faster processing
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()

    return output_predictions

def create_binary_mask(output_predictions, class_index):
    mask = output_predictions == class_index
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    return mask_image

def apply_mask(image_path, mask):
    image = Image.open(image_path).convert("RGB")
    masked_image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)
    return masked_image

def recombine_images(original_image_path, transformed_segment_path, mask_path):
    original_image = Image.open(original_image_path).convert("RGB")
    # original_image.show()
    mask = mask_path

    # mask = Image.open(mask_path).convert("RGB")

    # Resize the transformed segment to match the original image size
    transformed_segment = Image.open(transformed_segment_path).convert("RGB")

    recombined_image = Image.composite(transformed_segment, original_image, mask)

    return recombined_image
        
