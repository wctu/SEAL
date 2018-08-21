# Learning Superpixels with Segmentation-Aware Affinity Loss
[Learning Superpixels with Segmentation-Aware Affinity Loss](https://sites.google.com/site/wctu1009/cvpr18_superpixel) 

Wei-Chih Tu, Ming-Yu Liu, Varun Jampani, Deqing Sun, Shao-Yi Chien, Ming-Hsuan Yang, and Jan Kautz
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018

[Project](https://sites.google.com/site/wctu1009/cvpr18_superpixel) | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Tu_Learning_Superpixels_With_CVPR_2018_paper.html)

## Getting Started
In this repository, we provide the test code and the model trained on the BSDS500 dataset using the ERS algorithm as the superpixel segmenter. We also provide the evaluation scripts used in our experiments.

### Prerequisites
* Hardware: PC with NVIDIA GPU. We have tested the code with GeForce GTX 1080Ti and TitanXP.
* Software: CUDA 9.1, PyTorch 0.4.1, OpenCV 3.4.2

### Data Format
The superpixel labels are all integers, so we saved the superpixel labels as single-channel 16-bit png images.
We can read such png files using OpenCV imread() with extra -1 flag:
```
img = cv2.imread('input.png', -1)
```
In our experiments, we preprocess all the datasets so that the segmentation ground-truth maps are also in the same 16-bit png format.
In the ```/data``` folder we sample some examples from the BSDS500 test set for reference.

### Testing
Go to ```/test``` and run test.py
The ERSModule.so is a Python interface of the [ERS algorithm](https://github.com/mingyuliutw/EntropyRateSuperpixel).
We modify the original ERS algorithm a bit so that it can take pixel affinities as input. See readme_ERS.pdf for more details.

### Evaluation
Go to ```/eval``` and there are two python scripts for evaluation. 
To use the eval_par.py script, you will additionally need to install the ```joblib``` package to enable multi-threading.
It is particularly helpful when evaluating a large dataset along with many number of superpixels.
The core evaluation functions are written in C++. The file EvalSPModule.so is the Python interface of these functions.
See readme_eval.pdf for more details.

## Bibtex
If you find this repository useful in your research, please cite us:
```
@inproceedings{Tu-CVPR-2018,
    author = {Tu, Wei-Chih and Liu, Ming-Yu and Jampani, Varun and Sun, Deqing and Chien, Shao-Yi and Yang, Ming-Hsuan and Kautz, Jan},
    title = {Learning Superpixels with Segmentation-Aware Affinity Loss},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018},
}
```
