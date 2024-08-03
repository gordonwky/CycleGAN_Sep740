"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from image_segmentation import segment_image,  image_resize, create_binary_mask, apply_mask , recombine_images
from video_processing import extract_frames, frames_segment, frames_to_video


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    
    # Preprocessing part
    # Segment the video to frames 
    # Define the path
    opt = TestOptions().parse()  # get test options
    video_path = opt.dataroot
    # video_path = './VideoTransfer/horseriding.mp4'

    # Extract the word before ".mp4"
    word_before_mp4 = os.path.basename(video_path).split('.')[0]
    # segment the video to frames
    extract_frames_path = extract_frames_path = os.path.join('./VideoTransfer/frames/', word_before_mp4)
    extract_frames(video_path, extract_frames_path)

    # Image segmentation part
    # Load a pre-trained segmentation model
    segment_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    # 
    binary_folder , mask_folder = frames_segment(extract_frames_path, segment_model, word_before_mp4)



    # CycleGAN part

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # pass the segmented frame folder to opt.dataroot
    opt.dataroot = mask_folder
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # create the dataset with segmented image 



    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
    # Recombine the images
    web_dir +=  '/images'
    recombined_path = os.path.join(web_dir, 'transformed_frames')
    os.makedirs(recombined_path, exist_ok=True)
    for i in range(len(os.listdir(mask_folder))):
        image_path = os.path.join(extract_frames_path, f"frame_{i:05d}.jpg")
        transformed_segment_path = os.path.join(web_dir, f"frame_{i:05d}_fake.png")
        output = segment_image(image_path, segment_model)
        mask_image = create_binary_mask(output, class_index = 13)
        recombined = recombine_images(image_path, transformed_segment_path,mask_image)
        recombined.save(os.path.join(recombined_path, f"frame_{i:05d}.jpg"))
    
    frames_to_video(recombined_path, f'{word_before_mp4}_transformed.mp4')