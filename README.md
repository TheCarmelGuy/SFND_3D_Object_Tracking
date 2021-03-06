# SFND 3D Object Tracking

<img src="course_code_structure.png" width="779" height="414" />


## Description

This project does the following:

1. Finds Keypoints and Keypoint descriptors in image using several different classical CV algorithms 
2. Matches Keypoints from consecutive frames
3. Uses YOLO detector to find bounding boxes of vehicles in image
4. Project lidar points into image and clusters lidar points, based on bounding boxes 
5. Tracks vehicle directly in front of ego vehicle using keypoint matching matrix 
6. Calcultes Time-To-Collision using cluster lidar or keypoint descriptors with Constant Acceleration Model

## Report Results
To view results, look at the "results" directory. There you will find:

* ANALYSIS.md: goes over my observations of TTC computations for both the Camera case as well as the Lidar case
* ```<detector>-<descriptor>.csv:``` contains all the TTC results of each detector-descriptor pair. The first column represents TTC values as computed by Camera ans the second represents TTC values computed by Lidar


## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.
