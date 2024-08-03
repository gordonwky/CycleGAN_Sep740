
<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# CycleGAN with Image Segmentation 


The implementation of CycleGAN is adopted from [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesungp), and supported by [Tongzhou Wang](https://github.com/SsnL).


**Note**: The current software works well with PyTorch 1.4. 

If you use this code for your research, please cite:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)


Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib)


**We follow the Guidance below since we are using the pre-trained model**

Apply a pre-trained model (CycleGAN)
- You can download a pretrained model (e.g. horse2zebra) with the following script:
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```
- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh#L3) for all the available CycleGAN models.
- To test the model, you also need to download the  horse2zebra dataset:
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

- Then generate the results using
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```
- The option `--model test` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

## Test our video Transfer Dataset 

python video.py --dataroot ./VideoTransfer/horse_short.mp4 --name horse2zebra_pretrained --model test --no_dropout

## Folder introduction (Beside the original paper folder)

1. Video Transfer
Store testing MP4 and store the frames segmented. 

2. results
The transformed frames under the image folder store the transformed frame for the mp4 

## Python Files introduction (Beside the original paper folder)

1. image_segmentation.py
    define the function for performing image segmentation: 
    1. image\_resize: resize the input image to 256 $\times$ 256 
    2. segment\_image: segment the targeted part from the original image 
     3. create\_binary\_mask: masked the targeted part with 1 (white) and the background with 0 (black)  
     4. apply\_mask: composite the targeted image from original image to the masked part 
       5. recombine\_images composite the transformed image to the original image \\ \hline
         video\_processing.py & \item \\ \hline
         video.py & main program for our model: \\ \hline

2. video_processing.py
   define the function for performing video processing: 
      1. extract\_frames: extract the frames from mp4 
      2. frames\_segment: segments and process frame by frame 
      3. frames\_to\_video: write the transformed frames into mp4 

3. video.py:
    Similar to the original test.py: given a video and transfer to a transformed video.  

