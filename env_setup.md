# Environmnet Setup
OS : WSL Ubuntu ( [Reference Link on Windows](https://github.com/graphdeco-inria/gaussian-splatting/issues/332) )

## Cuda Toolkit
Make sure to install [cudatoolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) and update the environment path : 
```
echo 'export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```
After refreshing bash,  `nvcc --version` will show something like : 
```
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```

## Pytorch
* DO NOT RUN `conda env create --file environment.yml` !!!
* Set up your pytorch with :
```
conda create -n talking_gaussian python=3.7.13
conda activate talking_gaussian
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
* Make sure your cuda is working with : `python -c "import torch; print(torch.cuda.is_available());"`

## Submodules
```
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./gridencoder
```

## Other dependencies
```
pip install requirement.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
```
After completing all the steps above, you can now preprocess video data.

## OpenFace
* You have to install other dependencies such as `g++-8`, `opencv`, `dlib` and `OpenBLAS` first. You can install them step by step according to this [reference link](https://devjunhong.github.io/openface/how_to_install_openface/)
* if you encounter error installing `opencv-4.1.0`, try `opencv-4.10.0`
* If you encounter : `Package 'g++-8' has no installation candidate`, try [this](https://askubuntu.com/questions/1406962/install-gcc7-on-ubuntu-22-04)
* If you finally get to the last step, i.e. installing `OpenFace`, and you see the shit as below, go to `/usr/local/include/dlib/opencv/cv_image.h` and change `IplImage temp = img;` to `IplImage temp = cvIplImage(img);`
```
/usr/local/include/dlib/opencv/cv_image.h:37:22: error: conversion from ‘const cv::Mat’ to non-scalar type ‘IplImage’ {aka ‘_IplImage’} requested
             IplImage temp = img;
```

* I am really fucked up in this section cuz it took me  about 8hrs to debug.

# Some modifications
* `dataset_reader.py` : 110