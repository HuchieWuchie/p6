#include "img.h"

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <array>
#include <algorithm>


using namespace std;
using namespace cv;

Img::Img(int x_range, int y_range, std::vector<std::vector<std::vector<float>>> pointcloud_lines, std::vector<std::vector<float>> regions_of_interest){

    vector<vector<float>> matrix(y_range, vector<float>(x_range, 0));

    img = cv::Mat::zeros(cv::Size(x_range, y_range), CV_8UC1);

    //Compute ordered list of all y values
    vector<float> y_axis;
    for(int i = 0; i < int(pointcloud_lines.size()); i++){
      y_axis.push_back(pointcloud_lines[i][0][1]);
    }

    // Populate cv::Mat img with the regions of interest coordinates found.
    for(int i = 0; i < int(regions_of_interest.size()); i++){
      float z = regions_of_interest[i][2];
      int y_index = int(regions_of_interest[i][3]);
      int x_index = int(regions_of_interest[i][4]);
      matrix[int(regions_of_interest[i][3])][int(regions_of_interest[i][4])] = z;
      img.at<uchar>(y_index, x_index) = int(z*255);
    }

    // Threshold so every non-zero coordinate equals 255
    cv::threshold(img, img, 1, 255, cv::THRESH_BINARY );

    // Perform connectivity analysis
    vector<Vec4i> hierarchy;
    findContours( img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    //Compute centroid of connected blobs
    centroids = Img::compute_centroids(contours, 1000); //Computes centroids with pixel area larger 1000

    //Compute coordinates for left and right blob
    roi_coordinates = Img::compute_roi_coordinates(contours);
}

Img::~Img()
{
  //Dtor
}

vector<vector<vector<int>>> Img::compute_roi_coordinates(vector<vector<Point>> contours){

      //vector<Vec4i> hierarchy;

      vector<vector<vector<int>>> roi_coords(2);
      vector<int> coordinate;

      float left_centroid = centroids[0];
      float right_centroid = centroids[centroids.size()-1];
      vector<vector<Point>> roi(2);

      //Find the left and right blob
      for(int i = 0; i < int(contours.size()); i++){
        Moments mu = moments(contours[i], false);
        float m10 = mu.m10;
        float m00 = mu.m00;
        float cx = m10/m00;
        if(cx == left_centroid){
          roi[0] = contours[i];
          Mat drawing = Mat::zeros( img.size(), CV_8UC1 );
          drawContours( drawing, roi, 0, 255, -1, 8);
          flip(drawing, drawing, 1); // x axis
          //flip(drawing, drawing, 0); //y axis
          //imshow("left",drawing);
          //waitKey(0);

          for(int i = 0; i < drawing.rows; i++){
            for(int j = 0; j < drawing.cols; j++){
              if(int(drawing.at<uchar>(i,j)) == 255){
                coordinate.push_back(i); //y index
                coordinate.push_back(j); //x index
                roi_coords[0].push_back(coordinate); // left region of interest
                coordinate.clear();
              }
            }
          }
        }
        else if(cx == right_centroid){
          roi[1] = contours[i];
          Mat drawing = Mat::zeros( img.size(), CV_8UC1 );
          drawContours( drawing, roi, 1, 255, -1, 8);
          flip(drawing, drawing, 1); // x axis
          //flip(drawing, drawing, 0); //y axis
          //imshow("right",drawing);
          //waitKey(0);

          for(int i = 0; i < drawing.rows; i++){
            for(int j = 0; j < drawing.cols; j++){
              if(int(drawing.at<uchar>(i,j)) == 255){
                coordinate.push_back(i); //y index
                coordinate.push_back(j); //x index
                roi_coords[1].push_back(coordinate); // left region of interest
                coordinate.clear();
              }
            }
          }
        }
      }
      return(roi_coords);
}

vector<float> Img::compute_centroids(vector<vector<Point>> contours, int area_threshold = 1000){
  //Thresholds blobs smaller than given area_threshold and returns array of x centroids

  vector<float> centroid_x;
  for( int i = 0; i< int(contours.size()); i++ )
     {
       int area = contourArea(contours[i]); // Compute area of blob in pixels
       if (area > area_threshold){
         Moments mu = moments(contours[i], false);
         float m10 = mu.m10;
         float m00 = mu.m00;
         float cx = m10/m00; //Computing x centroid of blob
         centroid_x.push_back(cx);
       }
     }

  //sort x centroids in ascending order
  sort(centroid_x.begin(), centroid_x.end());
  return(centroid_x);

}

void Img::get_mask_lines(){

}
