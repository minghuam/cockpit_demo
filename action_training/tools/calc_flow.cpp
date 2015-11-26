#include <vector>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

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

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main(int argc, char** argv){

	const char* keys =
		{
			"{ p  | prevImg      | 001.jpg | filename of 1st image }"
			"{ n  | nextImg      | 002.jpg | filename of 2nd image }"
			"{ x  | xFlowFile    | flow_x.jpg | filename of flow x component }"
			"{ y  | yFlowFile    | flow_y.jpg | filename of flow y component }"
			"{ b  | bound | 20 | specify the maximum of optical flow}"
			"{ t  | type | 1 | specify the optical flow algorithm }"
			"{ d  | device_id    | 0  | set gpu id}"
			"{ s  | show    | 0  | show results}"	
		};

	CommandLineParser cmd(argc, argv, keys);
	string prevImg = cmd.get<string>("prevImg");
	string nextImg = cmd.get<string>("nextImg");
	string xFlowFile = cmd.get<string>("xFlowFile");
	string yFlowFile = cmd.get<string>("yFlowFile");
	int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");
    int show = cmd.get<int>("show");

	Mat cpu_frame_0 = imread(prevImg, IMREAD_GRAYSCALE);
	Mat cpu_frame_1 = imread(nextImg, IMREAD_GRAYSCALE);

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

	imwrite(xFlowFile, imgX);
	imwrite(yFlowFile, imgY);

	if(show > 0){
		imshow("1", cpu_frame_0);
		imshow("2", cpu_frame_1);
		imshow("X", imgX);
		imshow("Y", imgY);
		waitKey(0);
	}
}