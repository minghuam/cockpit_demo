#include <vector>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "directory.hpp"

using namespace cv;
using namespace cv::gpu;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
       double lowerBound, double higherBound) {
    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i,j);
            float y = flow_y.at<float>(i,j);
            img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
            img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
        }
    }
    #undef CAST
}

static void calc_one_flow(string prev_image, string next_image, 
                string flow_x_image, string flow_y_image, 
                int bound, int device_id, 
                int type){
    Mat cpu_frame_0 = imread(prev_image, IMREAD_GRAYSCALE);
    Mat cpu_frame_1 = imread(next_image, IMREAD_GRAYSCALE);

    setDevice(device_id);
    gpu::GpuMat gpu_frame_0, gpu_frame_1, gpu_flow_x, gpu_flow_y;
    FarnebackOpticalFlow alg_farn;
    OpticalFlowDual_TVL1_GPU alg_tvl1;
    BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

    gpu_frame_0.upload(cpu_frame_0);
    gpu_frame_1.upload(cpu_frame_1);

    switch(type){
        case 0:
            alg_farn(gpu_frame_0, gpu_frame_1, gpu_flow_x, gpu_flow_y);
            break;
        case 1:
            alg_tvl1(gpu_frame_0, gpu_frame_1, gpu_flow_x, gpu_flow_y);
            break;
        case 2:
            GpuMat d_frame_0f, d_frame_1f;
            gpu_frame_0.convertTo(d_frame_0f, CV_32F, 1.0 / 255.0);
            gpu_frame_1.convertTo(d_frame_1f, CV_32F, 1.0 / 255.0);
            alg_brox(d_frame_0f, d_frame_1f, gpu_flow_x, gpu_flow_y);
            break;
    }

    Mat cpu_flow_x, cpu_flow_y;
    gpu_flow_x.download(cpu_flow_x);
    gpu_flow_y.download(cpu_flow_y);

    Mat imgX(cpu_flow_x.size(), CV_8UC1);
    Mat imgY(cpu_flow_y.size(), CV_8UC1);

    convertFlowToImage(cpu_flow_x, cpu_flow_y, imgX, imgY, -bound, bound);

    imwrite(flow_x_image, imgX);
    imwrite(flow_y_image, imgY);
}

int main(int argc, char** argv){

    const char* keys =
        {
            "{ i  | input_folder       | folder | input image folder }"
            "{ x  | output_x_folder      | folder | output x flow folder }"
            "{ y  | output_y_folder      | folder | output y flow folder }"
            "{ b  | bound | 20 | specify the maximum of optical flow}"
            "{ t  | type | 1 | specify the optical flow algorithm }"
            "{ d  | device_id    | 0  | set gpu id}"
            "{ s  | show    | 0  | show results}"   
        };

    CommandLineParser cmd(argc, argv, keys);
    string input_folder = cmd.get<string>("input_folder");
    string output_x_folder = cmd.get<string>("output_x_folder");
    string output_y_folder = cmd.get<string>("output_y_folder");
    int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int show = cmd.get<int>("show");

    setDevice(device_id);
    gpu::GpuMat gpu_frame_0, gpu_frame_1, gpu_flow_x, gpu_flow_y;
    FarnebackOpticalFlow alg_farn;
    OpticalFlowDual_TVL1_GPU alg_tvl1;
    BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

    vector<string> images;
    fs::ls_files(images, input_folder, "jpg");
    
    Mat cpu_frame_0, cpu_frame_1;
    Mat cpu_flow_x, cpu_flow_y;
    
    for(int i = 0; i < images.size() - 1; i++){
        string basename = fs::basename(images[i]);
        string x_img = fs::join_path(output_x_folder, basename);
        string y_img = fs::join_path(output_y_folder, basename);
        
        if(i == 0){
            cpu_frame_0 = imread(images[i], IMREAD_GRAYSCALE);
        }else{
            cpu_frame_0 = cpu_frame_1;
        }
        cpu_frame_1 = imread(images[i+1], IMREAD_GRAYSCALE);

        gpu_frame_0.upload(cpu_frame_0);
        gpu_frame_1.upload(cpu_frame_1);

        switch(type){
            case 0:
                alg_farn(gpu_frame_0, gpu_frame_1, gpu_flow_x, gpu_flow_y);
                break;
            case 1:
                alg_tvl1(gpu_frame_0, gpu_frame_1, gpu_flow_x, gpu_flow_y);
                break;
            case 2:
                GpuMat d_frame_0f, d_frame_1f;
                gpu_frame_0.convertTo(d_frame_0f, CV_32F, 1.0 / 255.0);
                gpu_frame_1.convertTo(d_frame_1f, CV_32F, 1.0 / 255.0);
                alg_brox(d_frame_0f, d_frame_1f, gpu_flow_x, gpu_flow_y);
                break;
        }

        gpu_flow_x.download(cpu_flow_x);
        gpu_flow_y.download(cpu_flow_y);

        Mat imgX(cpu_flow_x.size(), CV_8UC1);
        Mat imgY(cpu_flow_y.size(), CV_8UC1);

        convertFlowToImage(cpu_flow_x, cpu_flow_y, imgX, imgY, -bound, bound);

        imwrite(x_img, imgX);
        imwrite(y_img, imgY);
    }
}
