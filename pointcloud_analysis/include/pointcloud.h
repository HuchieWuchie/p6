#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>



class Pointcloud
{


    public:
        Pointcloud();
        std::vector<std::vector<float>> from_txt(std::string filename);
        std::vector<std::vector<float>> delete_in_axis(std::vector<std::vector<float>> pointcloud, float value, int axis);
        std::vector<std::vector<std::vector<float>>> get_pointcloud_lines(std::vector<std::vector<float>> pointcloud);
        std::vector<float> get_column_vector(std::vector<std::vector<float>> matrix, int axis);
        std::vector<std::vector<float>> pcd_to_image(std::vector<std::vector<float>> pointcloud, std::vector<std::vector<std::vector<float>>> pointcloud_lines);
        std::vector<std::vector<float>> img_to_3d_grid(std::vector<std::vector<float>> matrix);
        std::vector<std::vector<float>> normalize(std::vector<std::vector<float>> pointcloud);
        void get_parameters(std::vector<std::vector<float>> pointcloud, std::vector<std::vector<std::vector<float>>> pointcloud_lines);
        void get_roi(int process_id, int stride);
        void split_pointcloud_lines();
        void combine_regions_of_interest();
        void save_to_txt(std::string filename);
        std::vector<std::vector<float>> mask(std::vector<std::vector<int>> mask, int border);
        virtual ~Pointcloud();

        //variables
        std::vector<std::vector<float>> regions_of_interest;
        std::vector<std::vector<std::vector<float>>> regions_of_interest_arr;
        std::vector<std::vector<float>> pcd;
        std::vector<std::vector<std::vector<float>>> pointcloud_lines;
        std::vector<std::vector<std::vector<std::vector<float>>>> pointcloud_lines_arr;

        cv::Mat img;

        float x_min, y_min, z_min;
        float x_max, y_max, z_max;
        int x_range, y_range;
        float x_resolution, y_resolution;

        torch::jit::script::Module net;


    protected:

    private:
};

#endif // POINTCLOUD_H
