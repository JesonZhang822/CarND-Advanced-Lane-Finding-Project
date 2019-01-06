# Advanced Lane Finding

Self-Driving Car NanoDegree

![output](./output_images/Output_image.png)

In this project, our goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want to create is a detailed writeup of the project.   

## Overview

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing our pipeline on single frames.  In order to extract more test images from the videos, we can simply use an image writing method like `cv2.imwrite()`, i.e., we can read the video in frame by frame as usual, and for frames we want to save for later we can write to an image file.  

The `challenge_video.mp4` video is an extra challenge to test our pipeline under somewhat trickier conditions.

## Writeup

### Step 0：Camera Calibration

Before using the camera，I  do the camera calibration:

* use chessboard images to obtain image points and object points
* use cv2.calibtateCamera() and cv2.undistort() to compute the calibration and undistortion

![images](./output_images/Undistorted.png)

### Step 1: Apply a distortion correction to test image

![images](./output_images/Undistorted_road.png)

### Step 2: Create thresholded binary image

Use color thretholds and gradient thresholds to find the lane lines in images :
* Color threshold : convert RGB color space to HLS color space,then thershold S chanel
* Gradient threshold : calculate the derivative in the x direction,then threshold x gradient

![images](./output_images/threshold_hls_sobel.png)

### Step 3:Perspective transform

Apply a perspective transform to rectify binary image :
* Identify four source points and destination points 
* Use cv2.getPerspectiveTransform() to calculate transform matrix
* Use cv2.warpPerspective() to warp the image

![images](./output_images/Prespectived_image.png)

### Step 4: Find lane boundary

Follow the next steps to find the lane boundary :

* Create histogram of binary image and find the peaks in the histogram
* Set the windows around the line centers and slide the windows
* Extract left and right line pixel positions
* Fit a polynomial

![images](./output_images/fit_poly.png)

###  Step 5 : Determine curvature  and vehicle position

After fitting a polynomial to those pixel positons,the [radius of curvature](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) of the fitting can be calculated:

* Calculate the radius of curvature based on pixel values
* Convert the x and y pixel values to real world space 
* Recalculate the radius of curvature
* Calculate the vehicle position with respect to center

### Step 6 : Warp back onto original image

![images](./output_images/Output_image.png)

### Step 7: Output video

After tuning on the test images,the pipeline will run on a video stream :

* Define a Line() class to keep track of the interesting parameters
* Input image frame by frame
* undistorted,prespected,threshold,ploy fit,calculate radius of curverad ...
* Sanity check : Checking the parallel and offset
* Look-Ahead Filter : search lane pixels around the previous detection
* Reset searching : if lossing the line,reset to use histogram and sliding window to search lane pixels. And use the predict fit as current fit
* Smoothing : average over the last 2 iterations

![images](./output_images/output_video.png)
