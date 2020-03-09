#include "pointcloud.h"

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <array>
#include <algorithm>
#include <fstream>


using namespace std;
using namespace cv;

Pointcloud::Pointcloud()
{

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    net = torch::jit::load("traced_model.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }

  regions_of_interest_arr = vector<vector<vector<float>>>(int(sysconf(_SC_NPROCESSORS_ONLN)));

}

Pointcloud::~Pointcloud()
{
    //dtor
}


vector<vector<float>> Pointcloud::from_txt(string filename){

    vector<float> point;
    vector<vector<float>> pointCloud;
    if(std::ifstream in {filename})
    {
        std::string coordinates;
        int i = 0;
        while(in >> coordinates)
        {
            float coordinate = strtof((coordinates).c_str(),0) ;
            if(i == 0){
                point.push_back(coordinate);
            }
            else if(i == 1){
                point.push_back(coordinate);

            }
            else if(i == 2){
                point.push_back(coordinate);
            }
            else if(i >= 2){
                i = -1;
                pointCloud.push_back(point);
                point.clear();
            }
            i++;
        }
    }
    else
    {
        cout << "Couldn't open file" << endl;
    }
    return(pointCloud);
}

vector<vector<float>> Pointcloud::delete_in_axis(vector<vector<float>> pointcloud, float value, int axis){

    vector<vector<float>> matrix;
    for(long unsigned int i = 0; i < pointcloud.size(); i++){
        if(pointcloud[i][axis] != value){
            matrix.push_back(pointcloud[i]);
        }
    }
    return(matrix);

}


vector<vector<vector<float>>> Pointcloud::get_pointcloud_lines(vector<vector<float>> pointcloud){

        /*
        Computes the point cloud lines as a numpy array. format [line][point][xyz]
        */
        vector<vector<float>> line;
        vector<vector<vector<float>>> pointcloud_lines;

        float holder = -1;
        for(long unsigned int i = 0; i < pointcloud.size(); i++){

            if(i == 0){
                holder = pointcloud[i][1];
                i++;
            }
            if(pointcloud[i][1] != holder){
                holder = pointcloud[i][1];
                pointcloud_lines.push_back(line);
                line.clear();
            }
            line.push_back(pointcloud[i]);

        }

        pointcloud_lines.push_back(line);

        Pointcloud::split_pointcloud_lines();

        return(pointcloud_lines);
}

void Pointcloud::split_pointcloud_lines(){

  pointcloud_lines_arr.clear();

  //multithread
  int numCPU = int(sysconf(_SC_NPROCESSORS_ONLN));
  if(numCPU > 1){
    int vector_size_big = pointcloud_lines.size() / (numCPU-1);
    int vector_size_small = pointcloud_lines.size() % (numCPU-1);

    for(int i = 0; i < numCPU; i++){
      if(i < numCPU-1){
        pointcloud_lines_arr.push_back(vector<vector<vector<float>>>(pointcloud_lines.begin() + vector_size_big*i, pointcloud_lines.begin() + vector_size_big*(i+1)));
      }
      else{
        pointcloud_lines_arr.push_back(vector<vector<vector<float>>>(pointcloud_lines.begin() + vector_size_small*i, pointcloud_lines.begin() + vector_size_small*(i+1)));
      }
    }
  }
}

vector<float> Pointcloud::get_column_vector(vector<vector<float>> matrix, int axis){
    /*
    Given a 2D matrix (vector of vector) returns column vector of given axis.
    */
    vector<float> column_vector;
    for(long unsigned int i = 0; i < matrix.size(); i++){
        column_vector.push_back(matrix[i][axis]);
    }
    return(column_vector);
}

void Pointcloud::get_parameters(vector<vector<float>> pointcloud, vector<vector<vector<float>>> pointcloud_lines){

    vector<float> x_axis = Pointcloud::get_column_vector(pointcloud, 0);
    vector<float> y_axis = Pointcloud::get_column_vector(pointcloud, 1);
    vector<float> z_axis = Pointcloud::get_column_vector(pointcloud, 2);

    x_min = *min_element(x_axis.begin(), x_axis.end());
    x_max = *max_element(x_axis.begin(), x_axis.end());
    y_min = *min_element(y_axis.begin(), y_axis.end());
    y_max = *max_element(y_axis.begin(), y_axis.end());
    z_min = *min_element(z_axis.begin(), z_axis.end());
    z_max = *max_element(z_axis.begin(), z_axis.end());

    y_range = pointcloud_lines.size();
    x_range = 0;

    for(int i = 0; i < y_range; i++){
        int line_length = pointcloud_lines[i].size()+1;
        if(line_length > x_range){
            x_range = line_length;
        }
    }

    x_resolution = (abs(x_min)+x_max)/x_range;
    y_resolution = (y_max-y_min)/(y_range-1);

}


vector<vector<float>> Pointcloud::pcd_to_image(vector<vector<float>> pointcloud, vector<vector<vector<float>>> pointcloud_lines){
        /*
        Returns a 2D matrix of the point cloud where all the values of the matrix
        are the z values
        */
        vector<vector<float>> matrix;
        vector<float> line;

        for(int i = 0; i < x_range; i++){
            line.push_back(x_max - (i*x_resolution));
        }

        for(int i = 0; i < y_range; i++){

            vector<float> z_values;
            vector<float> x_axis = Pointcloud::get_column_vector(pointcloud_lines[i], 0);
            std::reverse(x_axis.begin(),x_axis.end());

            for(int j = 0; j < x_range; j++){
                float x_value = line[j];

                auto it = std::lower_bound(x_axis.begin(), x_axis.end(), x_value);
                int index = it-x_axis.begin()-1;
                if (index == -1){
                    z_values.push_back(pointcloud_lines[i][index+1][2]);
                }
                else{
                    z_values.push_back(pointcloud_lines[i][index][2]);
                }
            }
            matrix.push_back(z_values);
            z_values.clear();

        }
        return(matrix);
}

vector<vector<float>> Pointcloud::img_to_3d_grid(vector<vector<float>> matrix){
    /*
        Returns a structured 3D grid [x, y, z] from a 2D matrix input [i, j] = z
    */
    vector<vector<float>> grid;
    for(int i = 0; i < y_range; i++){
        for(int j = 0; j < x_range; j++){

            float x = x_min+(j*x_resolution);
            float y = (i*y_resolution) + y_min;
            float z = matrix[i][j];
            vector<float> point;
            point.push_back(x);
            point.push_back(y);
            point.push_back(z);
            grid.push_back(point);
            point.clear();
        }
    }
    return(grid);

}


vector<vector<float>> Pointcloud::normalize(vector<vector<float>> pointcloud){
        /*
        All z-values are set normalized to between 0 and 1
        */
        for(long unsigned int i = 0; i < pointcloud.size(); i++){
            pointcloud[i][2] = (pointcloud[i][2] - z_min) / (z_max-z_min);
        }
        return(pointcloud);
}

void Pointcloud::get_roi(int process_id, int stride){

    int numCPU = int(sysconf(_SC_NPROCESSORS_ONLN));
    int vector_size_big = pointcloud_lines.size() / (numCPU-1);

    int y_offset = process_id*vector_size_big;


    vector<vector<float>> roi;
    vector<float> feature_vector;
    vector<float> point;
    for(int i = 0; i < int(pointcloud_lines_arr[process_id].size()); i++){
        for(int j = 0; j < int(pointcloud_lines_arr[process_id][i].size()); j = j + stride){
            if(j > 50 && j < int(pointcloud_lines_arr[process_id][i].size() - 51)){

                for(int k = 0; k < 101; k++){
                    feature_vector.push_back({pointcloud_lines_arr[process_id][i][j-50+k][2]});
                }

                auto input_tensor = torch::from_blob(feature_vector.data(), {1, 101});

                torch::Tensor output = net.forward({input_tensor}).toTensor();

                if(output[0][1].item<float>() > 0.8){
                  for(int k = 0; k < stride; k++){
                    point.push_back(pointcloud_lines_arr[process_id][i][j+k][0]);
                    point.push_back(pointcloud_lines_arr[process_id][i][j+k][1]);
                    point.push_back(pointcloud_lines_arr[process_id][i][j+k][2]);
                    point.push_back(i+y_offset); // row
                    point.push_back(j+k); // column
                    roi.push_back(point);
                    point.clear();
                  }
                }

                feature_vector.clear();

            }
        }
    }
    regions_of_interest_arr[process_id] = roi;
}

void Pointcloud::combine_regions_of_interest(){

    for(int i = 0; i < int(sysconf(_SC_NPROCESSORS_ONLN)); i++){
      regions_of_interest.insert(regions_of_interest.end(), regions_of_interest_arr[i].begin(), regions_of_interest_arr[i].end());

    }
}

vector<vector<float>> Pointcloud::mask(vector<vector<int>> mask, int border){

  vector<vector<int>> line;
  vector<vector<vector<int>>> pcd_lines;

  int holder = -1;
  for(int i = 0; i < int(mask.size()); i++){

      if(i == 0){
          holder = mask[i][0];
          i++;
      }
      if(mask[i][0] != holder){
          holder = mask[i][0];
          pcd_lines.push_back(line);
          line.clear();
      }
      line.push_back(mask[i]);

  }
  pcd_lines.push_back(line);

  //Append border

  vector<vector<float>> pcd;
  vector<float> point;

  for(int i = 0; i < int(pcd_lines.size()); i++){
    int x_index = pcd_lines[i][0][1];
    int y_index = pcd_lines[i][0][0];
    for(int j = x_index-border; j < x_index+int(pcd_lines[i].size()+border); j++){

      point.push_back(pointcloud_lines[y_index][j][0]); // x value
      point.push_back(pointcloud_lines[y_index][j][1]); // y value
      point.push_back(pointcloud_lines[y_index][j][2]); // z value
      pcd.push_back(point);
      point.clear();

    }
  }
  return(pcd);

}

void Pointcloud::save_to_txt(string filename){
  ofstream txt_file;
  txt_file.open(filename);
  for(int i = 0; i < int(pcd.size()); i++){
      txt_file << pcd[i][0] << " " << pcd[i][1] << " " << pcd[i][2] << endl;
  }
  txt_file.close();

}
