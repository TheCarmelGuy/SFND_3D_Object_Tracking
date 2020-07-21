
# Lidar TTC (FP 5)



* Sample 1: 

In frame 3 -> 4, the position of the car goes from: 7.85 -> 7.79 over 0.1 seconds. However the points selected were from 7.947 -> 7.891, which yields dramatically  different results:

Actual: 7.79*0.1/(7.85-7.79) = 12.983 
Real: 14.09 seconds

This most likely has to do with noise in the data. What TTC lidar is trying to do is use some metric of correspondence to say a lidar point in a previous frame is the same as the next. While median is more robst then the mean, it's still possible to have issues beacuse, we aren't guaranteed the same number points in consecutive frame. What's interesting about this particular frame is that there is a huge spike in the number of points, which mostlikely skews the results of using the median to use to develop correspondence between lidar points;



# Camera TTC (FP 6)


## Detector-Descriptor Model Selection

In order to consider how well the Camera TTC was, I used the Lidar TTC as the ground truth; Despite it's noisiness, Lidar TTC is going to be much more accurate than camera (due to it's ability to directly measure distance). With that, I use the norm values between lidar and camera ttc estimates to rank the preformance of descriptor+detector pairs. 

*** In this directory, you'll also notice a series of .csv files (labled as <detectior-name>-<descriptor-name>.csv), indicating the TTC values computed at each frame. The first column represents TTC from the Camera frame and the second corresonds to the corresponding TTC computed from the Lidar Data. *** 

AKAZE-AKAZE 6.5632
AKAZE-BRIEF 7.723
AKAZE-FREAK  7.6321
AKAZE-ORB 7.4189
AKAZE-SIFT 7.6978

BRISK-BRIEF 12.869
BRISK-FREAK 16.805
BRISK-ORB 13.686
BRISK-SIFT  19.715

FAST-BRIEF 6.5674
FAST-FREAK 8.2907
FAST-SIFT 5.9450
FAST-ORB 7.2511

ORB-BRIEF 182.42
ORB-FREAK 587.47
ORB-ORB 90.263
ORB-SIFT 106.60



SHITOMASI-BRIEF 9.3117
SHITOMASI-FREAK 9.6624
SHITOMASI-ORB 9.2614
SHITOMASI-SIFT 8.8365


SIFT-BRIEF 7.6888 
*SIFT-FREAK 5.0425
*SIFT-SIFT 5.6088

HARRIS-BRIEF 1399.1
HARRIS-FREAK = 9.8015
HARRIS-ORB 69.083
HARRIS-SIFT 634.25

Looking at these l2-norm values as well as being able to find the most matches from the frame-to-frame, it seems as though SIFT-FREAK and SIFT-SIFT are the best detector-discriptor pairs out of all of them.



## Things That Didn't work


Some interesting observations were that the harris detector generally didn't work as well. In general, it seemed like the HARRIS detector generated very few points and therefore caused bad matches. These bad matches cause erronous distance ratios, which propogate to TTC calculations. This might have been in better, if I used a better NMS algorithm for the harris detector



Another interesting observation is that the ORB detector didn't perform very well across the board, regardless of the the descriptor used. 
