## This repo is archived due to open source version of the original project will be uploaded and maintained.

### This file is used to track progress of face recognition part.

+ The prototype is still buggy and need to be fixed.
+ use command `python main.py` to run face recognition with siamese network prototype.
+ ~~Dataset available from https://github.com/StephenMilborrow/muct.git~~
+ New dataset available from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
+ Need to create a trainset/testset/validset csv/txt file before training which include names of all images.
+ File format as `IMAGE_NAME IMAGE_LABEL` in each line
+ Network contains 11 conv + 4 pooling + 2 maxout + 2 fc layers.

If program crashes, try to reduce `BATCH_SIZE`, require roughly 11G memory for BATCH_SIZE=130.

## TODO
+ Add validation step
+ Improving algorithm efficiency
+ ~~Maximize GPU usage~~
+ Add comment
+ ~~Add functionality to save and load trained model~~
+ Rewrite part of code for easy customizing
+ Rewrite according to given API

## Dependency
+ python 3.*
+ pytorch 0.4.1
+ torchvision
+ matplotlib
+ skimage
+ numpy
+ pandas



## Statement
+ The origin of this project is from one of my second year module in University of Nottingham.
+ That project focus on build a real time face recognition application for low quality video/CCTV.
+ This is face recognition part of that origin project, all writen by author.
