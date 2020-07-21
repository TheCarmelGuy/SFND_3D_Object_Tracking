#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {

        int normType = 0; 
        if(descriptorType.compare("DES_BINARY") ==0) 
        {
            normType = cv::NORM_HAMMING;
        }
        else if(descriptorType.compare("DES_HOG") == 0)
        {
            normType = cv::NORM_L2;
        }

        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
   
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation            descSource.convertTo(descSource, CV_32F);
        
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        
        double t = (double)cv::getTickCount(); 
        vector<vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource,descRef, knnMatches, 2);
            //knnMatch(descSource, vstd::vector<std::vector<DMatch> > &matches, int k)
     
        double minDescDistRatio = 0.8;
   
        for (auto it = knnMatches.begin(); it != knnMatches.end(); ++it)
        {
   
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        
        }
   
        std::cout<<"Thown out: "<<knnMatches.size() - matches.size()<<" matches wih knn"<<std::endl;


    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 3.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
  
        int numBytes = 32; 
        bool useOrientation = true;
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(numBytes, useOrientation);
    
    }
    else if(descriptorType.compare("ORB") == 0)
    {

        int features = 500;
        float scaleFactor = 2.0;
        int numLevels = 128;
        int edgeThreshold = 31;
        int firstLevel = 0;
        cv::ORB::ScoreType scoreType=cv::ORB::HARRIS_SCORE;
        int patchSize=31;
        int fastThreshold=20;


        extractor = cv::ORB::create(features, scaleFactor, numLevels, edgeThreshold,firstLevel, 2, scoreType,patchSize, fastThreshold);
   
    }
    else if(descriptorType.compare("FREAK") == 0)
    {

        bool  	orientationNormalized = true;
		bool  	scaleNormalized = true;
		float  	patternScale = 22.0f;
		int  	nOctaves = 4;
		const std::vector< int > &selectedPairs = std::vector< int >();
        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized,
                patternScale, nOctaves, selectedPairs);
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {

        extractor = cv::AKAZE::create();
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}


void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{


    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    int windowSize = 40; 
   
    double t = (double)cv::getTickCount();
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled,harris_keypoints;
    harris_keypoints = cv::Mat::zeros(img.size(), CV_32FC1);
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    
    for(int row = 0; row < dst.rows; row =row+windowSize)
    {
        for(int col = 0; col<dst.cols; col=col+windowSize)
        {
//            std::cout<<"row: "<<row<<" dist.rows: "<<dst.rows<<std::endl;
//            std::cout<<"col: "<<col<<" dist.cols: "<<dst.cols<<std::endl;
           
            //created crop

            int vertDist;
            if(row  + windowSize > dst.rows)
                vertDist =  static_cast<int>(dst.rows) - row;
            else 
                vertDist = windowSize;
            
            
            int horDist;
            if(col + windowSize > dst.cols)
                horDist = static_cast<int>(dst.cols) - col;
            else
                horDist = windowSize;


           // std::cout<<"vertDist: "<<vertDist<<" horDist: "<<horDist<<std::endl;
            cv::Rect crop(col,row,horDist, vertDist);
            cv::Mat maxSuppress = dst(crop); 
            //std::cout<<maxSuppress<<std::endl;
            //min and max point in crop
            double min, max;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(maxSuppress,&min,&max,&minLoc,&maxLoc); //find min and max of in neightbor
      

            maxLoc.x += col; // get location in original matrix
            maxLoc.y += row;
            
//            std::cout<<"Max: "<<max<<" x: "<<maxLoc.x<<" y: "<<maxLoc.y<<std::endl;
           
            // find max Suppressed 
            cv::KeyPoint maxSuppressCorner  = cv::KeyPoint(maxLoc.x, maxLoc.y, blockSize+5, -1, blockSize, 0,-1);
            keypoints.push_back(maxSuppressCorner); 
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "HARRIS detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;



   // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }



}





// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img,std::string detectorType, bool bVis)
{
       // Apply corner detection
    double t = (double)cv::getTickCount();



    if(detectorType.compare("FAST") == 0)
    {
        cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create();
        fastDetector->detect(img,keypoints, cv::Mat()); //assuming grayscle image
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;


    
    }
    else if(detectorType.compare("ORB") == 0)
    {

        cv::Ptr<cv::ORB> orbDetector = cv::ORB::create();
        orbDetector->detect(img,keypoints, cv::Mat()); //assuming grayscle image
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }

    else if(detectorType.compare("BRISK") == 0)
    {

        cv::Ptr<cv::BRISK> briskDetector = cv::BRISK::create();
        briskDetector->detect(img,keypoints, cv::Mat()); //assuming grayscle image
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }


    else if(detectorType.compare("AKAZE") == 0)
    {

        cv::Ptr<cv::AKAZE> akazeDetector = cv::AKAZE::create();
        akazeDetector->detect(img,keypoints, cv::Mat()); //assuming grayscle image
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if(detectorType.compare("SIFT") == 0)
    {

        cv::Ptr<cv::xfeatures2d::SIFT> siftDetector = cv::xfeatures2d::SIFT::create();
        siftDetector->detect(img,keypoints, cv::Mat()); //assuming grayscle image
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }


    


    

   // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}






// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
