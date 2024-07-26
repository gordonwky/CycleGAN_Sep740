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
    # image.show(title="Original Image")
    # mask = Image.fromarray((np.array(mask_image) / 255).astype(np.uint8))
    # mask.show(title="masked Image")
    # Apply the mask to the image
    masked_image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)
    # masked_image.show(title="Combined Image")
    return masked_image

# Assuming you have a function to load your CycleGAN model

# model = load_cyclegan_model()
# turn masked image into green 
def test_transform(masked_image):
    return masked_image
def recombine_images(original_image_path, transformed_segment, mask_image):
    original_image = Image.open(original_image_path).convert("RGB")
    original_image.show()
    mask = Image.fromarray((np.array(mask_image) / 255).astype(np.uint8))
    mask.show()
    # Resize the transformed segment to match the original image size
    transformed_segment = transformed_segment.resize(original_image.size, Image.BICUBIC)
    transformed_segment.show()
    # Recombine the transformed segment with the original image
    recombined_image = Image.composite(transformed_segment, original_image, mask)

    return recombined_image
        
    # webpage.save()  # save the HTML
# Apply the segmentation to an image
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

def frames_segment(image_folder,model):
    #transform the image in the folder
    for i in range(0, len(os.listdir(image_folder))):
        image_path = os.path.join(image_folder, f"frame_{i:05d}.jpg")
        output_predictions = segment_image(image_path, model)
        mask_image = create_binary_mask(output_predictions, class_index = 13)
        masked_image = apply_mask(image_path, mask_image)
        # masked_image.show()
        masked_image_folder = os.path.join('./VideoTransfer/masked_frames', f"frame_{i:05d}.jpg")
        masked_image.save(masked_image_folder)
    # Visualize the raw output




# video_path = './VideoTransfer/horseriding.mp4'
# extract_frames(video_path, './VideoTransfer/frames')

# model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# frames_segment('./VideoTransfer/frames',model)


# masked_image.show()
# transformed_segment = test_transform(masked_image)

# transformed_segment = transform_segmented_part(masked_image, cycle_transform, model)
# transformed_segment.show()

# # Recombine the images
for i in range(len(os.listdir('./VideoTransfer/masked_frames'))):
    image_path = os.path.join('./VideoTransfer/frames', f"frame_{i:05d}.jpg")
    masked_image = Image.open(os.path.join('./VideoTransfer/masked_frames', f"frame_{i:05d}.jpg"))
    transformed_segment_path = os.path.join('./results/horse2zebra_pretrained/test_latest/images', f"frame_{i:05d}_fake.png")
    transformed_segment = Image.open(transformed_segment_path).convert("RGB")
    mask_image = Image.open(os.path.join('./VideoTransfer/masked_frames', f"frame_{i:05d}.jpg"))
    recombined_image = recombine_images(image_path, transformed_segment, mask_image)

    # recombined_image.show()
# recombined_image = recombine_images(image_path, transformed_segment, mask_image)
# recombined_image.show()
# /Users/kimyingwong/CycleGAN_Sep740/results/horse2zebra_pretrained/test_latest/images/frame_00000_fake.png
