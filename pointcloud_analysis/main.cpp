#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <torch/script.h> // One-stop header.
#include <boost/date_time/posix_time/posix_time.hpp>
#include <thread>
#include <opencv2/opencv.hpp>
#include <fstream>


#include "pointcloud.h"
#include "img.h"
using namespace std;

//Not used right now
float curvature(float x, float x_1, float x_2, float step_size){
  float curvature = ((x_2 - (2*x_1)) + x) / pow(step_size,2);
  return(curvature);
}

vector<vector<float>> normalize(vector<vector<float>> vec, int kernel_size){
  vector<float> x_vector;

  for(int i = 0; i < int(vec.size()) + 2*kernel_size; i ++){
    if(i < kernel_size || i >= int(vec.size())){
      x_vector.push_back(0);
    }
    else{
      x_vector.push_back(vec[i][0]);
    }

  }

  for(int i = 0; i < int(x_vector.size()); i ++){
    if(i > kernel_size && i < int(vec.size())){
      vector<float> array;
      for(int j = i - kernel_size; j <= i + kernel_size; j++){
        array.push_back(x_vector[j]);
      }
      sort(array.begin(), array.end());
      float median = array.at(array.size() / 2);
      vec[i-kernel_size][0] = median;
      array.clear();
    }
  }
  x_vector.clear();

  return(vec);
}



string logger_filename = "log.txt";

int main(int argc, char** argv){
    ofstream logger;
    logger.open(logger_filename);


    //Evaluate filename string

    string filename;
    string argument;
    string save;
    if(argc <= 3){
      cout << "Define point cloud file and output (ex: ./program pointcloud.txt regions_of_interest save_true)" << endl;
      logger << boost::posix_time::microsec_clock::local_time() << ": ERROR: pointcloud argument missing." << endl;
      logger.close();
      return(0);
    }
    else if(argc == 4){
      filename = argv[1];
      argument = argv[2];
      save = argv[3];
      if(argument != "roi" && argument != "trajectories"){
        cout << "ERROR: Wrong output argument" << endl;
        logger << boost::posix_time::microsec_clock::local_time() << ": ERROR: Wrong output." << endl;
        logger.close();
        return 0;
      }
      if(save != "save_true" && save != "save_false"){
        cout << "ERROR: Wrong save argument" << endl;
        logger << boost::posix_time::microsec_clock::local_time() << ": ERROR: Wrong output." << endl;
        logger.close();
        return 0;
      }
      cout << filename << endl;
      logger << boost::posix_time::microsec_clock::local_time() << ": INFO: Pointcloud: " << filename << " OUTPUT: " << argv[2] << endl;
    }
    else{
      cout << "Define point cloud file and output (ex: ./program pointcloud.txt regions_of_interest)" << endl;
      logger << boost::posix_time::microsec_clock::local_time() << ": ERROR: too many input arguments." << endl;
      logger.close();

      return(0);
    }

    // Load point cloud
    Pointcloud pointcloud;
    pointcloud.pcd = pointcloud.from_txt(filename);
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, Loaded pointcloud." << endl;
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, Pointcloud size: " << pointcloud.pcd.size() << endl;
    logger << endl << endl;

    // Remove zero points
    pointcloud.pcd = pointcloud.delete_in_axis(pointcloud.pcd, 0, /*axis=*/ 2);
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, zero points removed." << endl;
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, Pointcloud size: " << pointcloud.pcd.size() << endl;

    // Initialize point cloud parameters
    pointcloud.pointcloud_lines = pointcloud.get_pointcloud_lines(pointcloud.pcd);
    pointcloud.get_parameters(pointcloud.pcd, pointcloud.pointcloud_lines);
    logger << endl << endl;
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, pointcloud paramters: " << endl;
    logger << "x minimum:" << "\t" << pointcloud.x_min << endl;
    logger << "x maxmimum:" << "\t" << pointcloud.x_max << endl;
    logger << "y minimum:" << "\t" << pointcloud.y_min << endl;
    logger << "y maxmimum:" << "\t" << pointcloud.y_max << endl;
    logger << "z minimum:" << "\t" << pointcloud.z_min << endl;
    logger << "z maxmimum:" << "\t" << pointcloud.z_max << endl;
    logger << "x resolution: " << "\t" << pointcloud.x_resolution << endl;
    logger << "y resolution: " << "\t" << pointcloud.y_resolution << endl;

    logger << "x range: " << "\t" << pointcloud.x_range << endl;
    logger << "y range: " << "\t" << pointcloud.y_range << endl;
    logger << endl << endl;

    // Normalize and re calculate parameters
    pointcloud.pcd = pointcloud.normalize(pointcloud.pcd);
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, normalized pointcloud." << endl;
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, normalized pointclod size: " << pointcloud.pcd.size()<< endl;
    if(save == "save_true"){
      pointcloud.save_to_txt("Normalized.txt");
    }
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, saved normalized pointcloud." << endl;
    pointcloud.pointcloud_lines = pointcloud.get_pointcloud_lines(pointcloud.pcd);

    // Construct grid, aka structured pointcloud.
    vector<vector<float>> img = pointcloud.pcd_to_image(pointcloud.pcd, pointcloud.pointcloud_lines);
    pointcloud.pcd = pointcloud.img_to_3d_grid(img);
    if(save == "save_true"){
      pointcloud.save_to_txt("grid.txt");
    }
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, pointcloud to grid." << endl;
    pointcloud.pointcloud_lines = pointcloud.get_pointcloud_lines(pointcloud.pcd);

    // Compute region of interest
    int numCPU = int(sysconf(_SC_NPROCESSORS_ONLN));
    std::thread th[numCPU];

    for(int i = 0; i < numCPU; i++){
      th[i] = std::thread(&Pointcloud::get_roi, &pointcloud, /* process_id */i, /* stride */5);
    }
    for(int i = 0; i < numCPU; i++){
      th[i].join();
    }
    pointcloud.combine_regions_of_interest();
    logger << boost::posix_time::microsec_clock::local_time() << ": INFO, region of interest size: " << pointcloud.regions_of_interest.size() << endl;
    Pointcloud ROI;
    ROI.pcd = pointcloud.regions_of_interest;
    if(save == "save_true"){
      ROI.save_to_txt("regions_of_interest.txt");
    }
    // Perform connectivity analysis
    Img im = Img(pointcloud.x_range, pointcloud.y_range, pointcloud.pointcloud_lines, pointcloud.regions_of_interest);
    vector<vector<vector<float>>> weld_seam;
    for(int i = 0; i < int(im.roi_coordinates.size()); i++){
      weld_seam.push_back(pointcloud.mask(im.roi_coordinates[i], /* border = */ 50));
    }

    // Save weld seams
    if(argument == "roi"){
      
      // Upscale pointcloud (unnormalize)
 
      for(int i = 0; i < int(weld_seam.size()); i++){
        for(int j = 0; j < int(weld_seam[i].size()); j++){
            float z_norm = weld_seam[i][j][2];
            weld_seam[i][j][2] = (z_norm * pointcloud.z_max) - (z_norm * pointcloud.z_min) + pointcloud.z_min;
        }
      }
      
      for(int i = 0; i < int(weld_seam.size()); i++){
        Pointcloud pcd_weld_seam;
        pcd_weld_seam.pcd = weld_seam[i];

        if(i == 0){
          pcd_weld_seam.save_to_txt("weldseam0.txt");
        }
        else if(i == 1){
          pcd_weld_seam.save_to_txt("weldseam1.txt");
        }
      }
      logger.close();
      return 0;
    }

    //Compute trajectories
    if(argument == "trajectories"){

      vector<vector<vector<float>>> trajectories;
      for(int u = 0; u < int(weld_seam.size()); u++){
        torch::jit::script::Module net;
        net = torch::jit::load("traced_model.pt");
        Pointcloud pcd_weld_seam;
        pcd_weld_seam.pcd = weld_seam[u];


        pcd_weld_seam.pointcloud_lines = pointcloud.get_pointcloud_lines(pcd_weld_seam.pcd);

        vector<float> point;
        vector<vector<float>> trajectory;
        for(int i = 0; i < int(pcd_weld_seam.pointcloud_lines.size()); i++){
          float max_prediction = 0;
          //float max_curvature = 0;
          //int step_size = 2;
            for(int j = 0; j < int(pcd_weld_seam.pointcloud_lines[i].size()); j++){
                if(j > 50 && j < int(pcd_weld_seam.pointcloud_lines[i].size() - 50)){
                //if(j > step_size && j < int(pcd_weld_seam.pointcloud_lines[i].size() - 2*step_size)){

                  vector<float> feature_vector;

                    for(int k = 0; k < 101; k++){
                      float pcd_point = pcd_weld_seam.pointcloud_lines[i][j-50+k][2];
                      feature_vector.push_back(pcd_point);
                    }

                    auto input_tensor = torch::from_blob(feature_vector.data(), {1, 101});
                    torch::Tensor output = net.forward({input_tensor}).toTensor();

                    //float curv = curvature(pcd_weld_seam.pointcloud_lines[i][j][2], pcd_weld_seam.pointcloud_lines[i][j+step_size][2], pcd_weld_seam.pointcloud_lines[i][j+2*step_size][2], float(step_size));

                    if(output[0][1].item<float>() > max_prediction){
                    //if(curv > max_curvature){
                      max_prediction = output[0][1].item<float>();
                      //max_curvature = curv;
                      point.clear();
                      point.push_back(pcd_weld_seam.pointcloud_lines[i][j][0]);
                      point.push_back(pcd_weld_seam.pointcloud_lines[i][j][1]);
                      //point.push_back((pcd_weld_seam.pointcloud_lines[i][j][2] * pointcloud.z_max) - (pcd_weld_seam.pointcloud_lines[i][j][2]*pointcloud.z_min) + pointcloud.z_min);
                      point.push_back(pcd_weld_seam.pointcloud_lines[i][j][2]);
                    }

                    feature_vector.clear();
                }
            }
            trajectory.push_back(point);
            point.clear();

        }
        trajectories.push_back(trajectory);
        trajectory.clear();

      }

      // Treat x values to a median outlier removal
      int kernel_size = 2;
      for(int i = 0; i < int(trajectories.size()); i++){
          trajectories[i] = normalize(trajectories[i], kernel_size);
          for(int j = 0; j < int(trajectories[i].size()); j++){
        }
      }

      // Upscale pointcloud (unnormalize)

      for(int i = 0; i < int(trajectories.size()); i++){
        for(int j = 0; j < int(trajectories[i].size()); j++){
          float z_norm = trajectories[i][j][2];
          trajectories[i][j][2] = (z_norm * pointcloud.z_max) - (z_norm * pointcloud.z_min) + pointcloud.z_min;
        }
      }

      // Save trajectories to txt

      for(int i = 0; i < int(trajectories.size()); i++){
        ofstream txt_file;
        txt_file.open("trajectory_" + to_string(i) + ".txt");
        for(int j = 0; j < int(trajectories[i].size()); j++){
          txt_file << trajectories[i][j][0] << " " << trajectories[i][j][1] << " " << trajectories[i][j][2] << endl;
        }
        txt_file.close();
      }
    }
    logger.close();
    return 0;
}
