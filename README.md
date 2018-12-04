### This file is used to track the progress.

+ This project will be synchronized with the origin face recognition of origin project.
+ Trained model will be saved as trainedmodel.pt.
+ Use `runTest()` function for testing model.
+ ~~Dataset available from https://github.com/StephenMilborrow/muct.git~~
+ New dataset available from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
+ Need to create a trainset/testset/validset csv/txt file before training which include names and labels of images.
+ File format as `IMAGE_NAME IMAGE_LABEL` in each line

If program crashes, try to reduce `BATCH_SIZE`.

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
