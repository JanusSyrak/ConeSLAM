# ConeSLAM

All code in this project, unless otherwise stated, has been written by Stefan Bjørn Gunnarsson and Janus Nørgaard Syrak Hansen. The project was developed in order to solve the SLAM problem in a Formula Student environment. 

The data used in this project can be found on the following website: https://nextcloud.sdu.dk/index.php/s/oCaAxmEwjd9LTEP

This both includes images along with their labelled masks, and point cloud observations and images from a test drive.

The code is divided into three different folders:

ConeCenterEstimation estimates the centers of cones based on the point cloud. 

ConeSegmentation includes both training and prediction of several neural networks to perform semantic segmentation of cones. 

ConeSLAM is our implementation of FastSLAM 1.0 and FastSLAM 2.0, and is the primary focus of our thesis. 

The code has been written to run on our local machines, and so, requires some modification to run. If you require any files or tips on how to use, please let me know. 
