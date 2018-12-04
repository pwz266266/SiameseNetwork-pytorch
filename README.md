### This file is used to track progress of face recognition part.

+ The prototype is still buggy and need to be fixed.
+ use command `python main.py` to run face recognition with siamese network prototype.
+ ~~Dataset available from https://github.com/StephenMilborrow/muct.git~~
+ New dataset available from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
+ Need to create a trainset/testset/validset csv/txt file before training which include names of all images.
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
