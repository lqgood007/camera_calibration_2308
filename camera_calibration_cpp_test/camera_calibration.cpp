#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
int main1()
{
    cv::Mat RawImage = cv::imread("../1.jpg");
    cv::namedWindow("RawImage",cv::WINDOW_NORMAL);
    cv::namedWindow("UndistortImage",cv::WINDOW_NORMAL);


    const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 455.47885132,   0.       ,  341.88040785  , 0.      ,   454.57400513,
 235.71359661 ,  0.     ,      0.       ,    1.       );
    const cv::Mat D = ( cv::Mat_<double> ( 1,5) << -2.78497190e-01 , 3.85005505e-02 , 1.42750506e-03 ,-1.65970830e-04,
  1.00151349e-01);

    cv::Mat map1,map2;
    cv::Size imageSize(RawImage.cols,RawImage.rows);
    const double alpha = 1;
    cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);
    initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);

    cv::Mat UndistortImage;
    remap(RawImage, UndistortImage, map1, map2, cv::INTER_LINEAR);


    cv::imshow("UndistortImage", UndistortImage);
    cv::imshow("RawImage",RawImage);
    cv::waitKey(0);
    return 0;
}