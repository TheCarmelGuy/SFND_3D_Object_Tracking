#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
    
    
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-60, bottom+10), cv::FONT_ITALIC, 0.4, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-60, bottom+30), cv::FONT_ITALIC, 0.4, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{

    //push back all current key points
    for(auto itr =  kptsCurr.begin(); itr != kptsCurr.end() ; itr++)
    {
        if(boundingBox.roi.contains(itr->pt))
        {
            boundingBox.keypoints.push_back(*itr);
        }
    }


  
    //--
    //COMPUTE MEAN DISTANCE OF MATCHES
    double meanDist = 0;
    for(auto itr = kptMatches.begin(); itr != kptMatches.end(); itr++)
    {
        cv::KeyPoint currKpt = kptsCurr[itr->trainIdx];
        cv::KeyPoint prevKpt = kptsPrev[itr->queryIdx];
        if(boundingBox.roi.contains(currKpt.pt))
        {
       
            meanDist += cv::norm(currKpt.pt - prevKpt.pt);
        }
    }
    meanDist = meanDist/kptMatches.size();
 
    //-- 
    //POPULATE KPTS ONLY W/ DISTANCES BELOW MEAN THRESH
    //--
    for(auto itr = kptMatches.begin(); itr != kptMatches.end(); itr++)
    {
        cv::KeyPoint currKpt = kptsCurr[itr->trainIdx];
        cv::KeyPoint prevKpt = kptsPrev[itr->queryIdx];
     
        double distVal =  cv::norm(currKpt.pt - prevKpt.pt);  

        double threshVel = fabs(fabs(distVal) - fabs(meanDist));
        if(boundingBox.roi.contains(currKpt.pt) &&
                threshVel < 20.0f)
        {
            boundingBox.kptMatches.push_back(*itr);
        }
    }
    

}


// Compute tme-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
  
    //--
    //GEN DIS RATIOS
    //--
    std::cout<<"INFO: Generating Dist Ratios"<<std::endl;
    std::vector<double> distRatios; 
    for(auto matchOuterItr = kptMatches.begin() ; matchOuterItr != kptMatches.end()-1; matchOuterItr++)
    {

        cv::KeyPoint prevOuterKpt = kptsPrev.at(matchOuterItr->queryIdx); 
        cv::KeyPoint currOuterKpt = kptsCurr.at(matchOuterItr->trainIdx); 
       
   
        for(auto matchInnerItr = kptMatches.begin() + 1; matchInnerItr != kptMatches.end(); matchInnerItr ++)
        {

            cv::KeyPoint prevInnerKpt = kptsPrev.at(matchInnerItr->queryIdx); 
            cv::KeyPoint currInnerKpt = kptsCurr.at(matchInnerItr->trainIdx); 
      

            
            double distPrev = cv::norm(prevOuterKpt.pt -  prevInnerKpt.pt);
            double distCurr = cv::norm(currOuterKpt.pt - currInnerKpt.pt);
          
            if(distPrev > std::numeric_limits<double>::epsilon() &&
                    distCurr >= 100.0) // ensure that key point is far apart and that we're not dividing by 0
            {
    
                double distRatio = distCurr/distPrev;
              //  std::cout<<"DEBUG: Following Dist value: "<<distRatio<<std::endl;

                distRatios.push_back(distRatio);
          
            }
        }

    }


    //----
    //PURNE OUTLIERS 
    //----


    double meanDistRatio = std::accumulate(distRatios.begin(),distRatios.end(), 0.0)/distRatios.size();

    distRatios.erase(std::remove_if(distRatios.begin(), distRatios.end(), 
                [&](double value){

                double thresh = fabs(fabs(value) - meanDistRatio);
    
                return (thresh> 0.05);

                }), distRatios.end());

        
        
    meanDistRatio = std::accumulate(distRatios.begin(),distRatios.end(), 0.0)/distRatios.size();

    std::cout<<"MEAN: "<<meanDistRatio<<std::endl;

    //---
    //SELECT POINT
    //---

    std::cout<<"INFO: Selecting Dist Ratio"<<std::endl;
     std::sort(distRatios.begin(), distRatios.end());

     int medianDistRatioIdx = std::floor(distRatios.size()/2);
     double medianDistRatio= distRatios.size() % 2 == 0  ? (distRatios[medianDistRatioIdx] + distRatios[medianDistRatioIdx - 1])/2.0f : distRatios[medianDistRatioIdx];
   

     

     if(medianDistRatio == 1.0f) //if for some reason we have a median of 1, then our formula doesn't work and use the mean instead
     {
       medianDistRatio = meanDistRatio;
     }
    

   
     double dT = 1.0f/frameRate;
     double medianRatioDenom = (1.0f - medianDistRatio);
   
     
   

     TTC = (-1.0f * dT)/medianRatioDenom;


}



bool compareLidarDist(const LidarPoint& lhs, const LidarPoint& rhs)
{
    return lhs.x < rhs.x;

}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{

   
    /*
     * Algorithm Overview
     * 1. Find Median Points (Robust to outliers) or prev and current frame
     * 2. Use located points to find computer TTC using TTC = d_1*dt/(d_0 - d_1)
     */

    double dT = 1.0f/frameRate;

    // find median  

    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), compareLidarDist);
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), compareLidarDist);
 
    double medPrev = lidarPointsPrev[lidarPointsPrev.size()/2].x;
    double medCurr = lidarPointsCurr[lidarPointsCurr.size()/2].x;



    std::cerr<<"Median Point:"<<lidarPointsCurr[lidarPointsCurr.size()/2].x<<std::endl;
    std::cerr<<"Median Point Prev:"<<lidarPointsPrev[lidarPointsPrev.size()/2].x<<std::endl;
    std::cerr<<"dT:"<<dT<<std::endl;

  //  compute TTC from both measurements
   TTC = medCurr * dT / (medPrev-medCurr);

}



void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{



       
    cv::Mat matchesMatrix(cv::Mat::zeros(prevFrame.boundingBoxes.size(),currFrame.boundingBoxes.size(),cv::DataType<int>::type));
    std::multimap<int,cv::DMatch> previousBoundingBoxMatches;
    std::multimap<int, cv::DMatch> currentBoundingBoxMatches;


    /*
     * MATCHING ALGORITHM
     * 1. BBKeypoints: Find keypoints in each bouding box for both previous frame and current frame, P, C 
     * 2. Matching Matrix For each bounding box, p_i, find matches in c_i and update match matrix to reflect number of matches it p_i, that match to c_i
     * 3. Graph Match: Find optimal mapping w/ matrix
     */

   
    //BB Keypoints O(2n^2)
    for(auto matchItr = matches.begin(); matchItr < matches.end(); matchItr ++)
    {
        //find all bounding boxes associated to matched points in previous and current frame
        for(auto prevFrameItr = prevFrame.boundingBoxes.begin();  prevFrameItr < prevFrame.boundingBoxes.end() ; ++ prevFrameItr)
        {

        //    cout<<"Num Lidar: "<<prevFrame.lidarPoints.size()<<std::endl;
            cv::KeyPoint prevMatchKeypoint = prevFrame.keypoints[matchItr->queryIdx];
           
//            std::cout<<"Key Point: "<<prevMatchKeypoint.pt<<std::endl;
            if(prevFrameItr->roi.contains(prevMatchKeypoint.pt))
            {
                previousBoundingBoxMatches.insert(std::pair<int, cv::DMatch>(prevFrameItr->boxID,*matchItr));
            }
        }
        //find all bounding boxes associated to matched points in previous and current frame
        for(auto currFrameItr = currFrame.boundingBoxes.begin();  currFrameItr < currFrame.boundingBoxes.end() ; ++ currFrameItr)
        {
            cv::KeyPoint currMatchKeypoint = currFrame.keypoints[matchItr->trainIdx];
            if(currFrameItr->roi.contains(currMatchKeypoint.pt))
            {

                currentBoundingBoxMatches.insert(std::pair<int, cv::DMatch>(currFrameItr->boxID,*matchItr));
            }
        }
    }




    //Generate Match Matrix O(n^2)
    for(auto prevBoundingBoxesItr = previousBoundingBoxMatches.begin(); prevBoundingBoxesItr !=  previousBoundingBoxMatches.end(); prevBoundingBoxesItr++)
    {
        int previousId = prevBoundingBoxesItr->first; 
        cv::DMatch prevMatch = prevBoundingBoxesItr->second;
        for(auto currBoundingBoxesItr = currentBoundingBoxMatches.begin(); currBoundingBoxesItr != currentBoundingBoxMatches.end(); currBoundingBoxesItr++)
        {

        
            int currId = currBoundingBoxesItr->first;
            cv::DMatch currMatch = currBoundingBoxesItr->second;
   
            if(currMatch.queryIdx == prevMatch.queryIdx)
            {
                matchesMatrix.at<int>(previousId, currId) =  matchesMatrix.at<int>(previousId, currId) + 1;
            }
        }
    }




    std::cout<<matchesMatrix<<std::endl;


    //Graph-match: Greedyly match each prev detection to current detection 
    //O(n^2) 
    //TODO: Come back to this. Can probably make a better policy for matching 
    for(int rowNum = 0; rowNum < matchesMatrix.rows; rowNum ++)
    {
        int prevId = rowNum;
        cv::Mat rowvec(matchesMatrix.row(rowNum)); // reg row vec

        cv::Point currId;
        cv::Point buff;
        double minMatches, maxMatches;
        cv::minMaxLoc(rowvec,&minMatches, &maxMatches,&buff, &currId,cv::noArray()); 

        //if there are too few matches, disregard 
        if(static_cast<int>(maxMatches) > 10)
        { 
            std::pair<int, int> bbMapping = std::pair<int,int>(prevId, currId.x); 
            std::cout<<prevId<<"->"<<currId.x<<" matches: "<<maxMatches<<std::endl;
            //zero out row vec to force unique mappings
            cv::Mat zeroColVec(cv::Mat::zeros(matchesMatrix.rows, 1, cv::DataType<int>::type));
            zeroColVec.copyTo(matchesMatrix.col(currId.x)); // reg row vec
     
            bbBestMatches.insert(bbMapping);//push back best result
            // std::cout<<matchesMatrix<<std::endl;
       
        }
    }


}





