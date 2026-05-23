//
// Created by moinshaikh on 3/4/26.
//


#include<iostream>
#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include"../../external/third_party/doctest.hpp"

void drawCircle(const cv::Mat &image)
{
    cv::Point center(image.cols / 2, image.rows / 2);
    int radius = 100;
    cv::Scalar color(0, 255, 0); // green
    int thickness = 2;
    int lineType = cv::LINE_AA;
    cv::circle(image, center, radius, color, thickness, lineType);
}

void drawRectangle(const cv::Mat &image) {

    cv::Point center(image.cols / 2,image.rows /2 );
    cv::Point center2(image.cols / 3 ,image.rows /3 );
    cv::Scalar colour(0,0,255);
    int thickness = 2;
    int lineType  = cv::LINE_AA;
    cv::rectangle(image,center,center2,colour, thickness,lineType);
}
TEST_CASE("ReadingImages")
{


}