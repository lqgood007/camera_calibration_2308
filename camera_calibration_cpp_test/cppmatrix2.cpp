#include<iostream>
#include<vector>
#include<string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/highgui.h"

using namespace std;
using namespace cv;

int main(int argc,char** argv) {
    cout<<"OpenCv Version: "<<CV_VERSION<<endl;
    Mat img=imread("../2.jpg");
    if(img.empty()){
        cout << "请确认图像文件名称是否正确" << endl;
        return -1;
    }
    //通过Image Watch查看的二维码四个角点坐标
// 1.16369722e+00  1.76467880e+00 -8.16900090e+02]
//  [-5.61313305e-01  2.64345967e+00  1.91498069e+02]
//  [ 1.68156164e-04  1.91673285e-03  7.85649127e-01]]

    const cv::Mat M = ( cv::Mat_<double> ( 3,3 ) << 1.16369722e+00,  1.76467880e+00 ,-8.16900090e+02,
    -5.61313305e-01 , 2.64345967e+00 , 1.91498069e+02 ,1.68156164e-04  ,1.91673285e-03 , 7.85649127e-01 );

    Mat img_warp;
    cv::warpPerspective(img,img_warp,M,img.size());
    imshow("img",img);
    imshow("img_warp",img_warp);
    waitKey(0);
    return 0;
}
