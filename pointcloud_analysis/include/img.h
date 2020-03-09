#ifndef IMG_H
#define IMG_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>



class Img
{


    public:
        Img(int x_range, int y_range, std::vector<std::vector<std::vector<float>>> pointcloud_lines, std::vector<std::vector<float>> regions_of_interest);
        //Img();
        virtual ~Img();

        std::vector<float> compute_centroids(std::vector<std::vector<cv::Point>> contours, int area_threshold = 1000);
        std::vector<std::vector<std::vector<int>>> compute_roi_coordinates(std::vector<std::vector<cv::Point>> contours);
        void get_mask_lines();


        //variables

        cv::Mat img;
        std::vector<float> centroids;
        std::vector<std::vector<std::vector<int>>> roi_coordinates;
        std::vector<std::vector<cv::Point>> contours;
        //std::vector<std::vector<std::vector<float>>> pointcloud_lines;

    protected:

    private:
};

#endif // IMG_H
