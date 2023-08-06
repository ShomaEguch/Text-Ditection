#pragma once

#include <iostream>
//#include <sstream>
//#include <fstream>
#include <vector>
#include <ctime>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#pragma comment(lib, "opencv_core247.lib")
#pragma comment(lib, "opencv_imgproc247.lib")
#pragma comment(lib, "opencv_ml247.lib")
#pragma comment(lib, "opencv_highgui247.lib")

using namespace std;

#include "Labeling.h"
#include "ConnectedComp.h"

bool GTCheck100(IplImage *gt_img, IplImage *mask_img);

#define MODE (1)
#define VECTOR_DIM (12)
#define WINDOW_RANGE (5)

typedef vector<double>::size_type vc_sz;

int main(){

	CvRTrees trees;
	trees.load("/path/to/classifier/file");

	for(int k = 0 ; k < 150 ; k++){
		//ofstream ofs1, ofs2;
		char input[512], input_color[512], input_gray[512], output[512];
		char filename[512];


		IplImage *src_img;
		IplImage *src_color, *src_gray;
		//ground truth directory
		// To directory where NIBLACK results exist
	    sprintf(input, "C:\\imagedata\\niblack\\dataset2_150_plus_GT_all_character\\image%d.bmp", k+1);
		// To original color image
		sprintf(input_color, "C:\\imagedata\\dataset2_150\\image%d.jpg", k+1);
		// To original color image, but read as gray image
		sprintf(input_gray, "C:\\imagedata\\dataset2_150\\image%d.jpg", k+1);
		// To directory where feature file will be saved
		sprintf(filename, "C:\\imagedata\\textdataplus\\dataset2_150_cc12\\image%d.txt", k+1);
		//sprintf(filename, "C:\\imagedata\\textdataplus\\dataset2_150_allfeature\\image%d.txt", k+1);

		sprintf(output, "/path/to/result/image.png", k+1);

		// IplImage data loading
		src_img = cvLoadImage(input, CV_LOAD_IMAGE_GRAYSCALE);
		src_color = cvLoadImage(input_color, CV_LOAD_IMAGE_ANYCOLOR);
		src_gray = cvLoadImage(input_gray, CV_LOAD_IMAGE_GRAYSCALE);
		if(!src_img || !src_color ){
			std::cout << "Not Found BMP Image Data!" << endl;
			continue;
		}

		
		cv::Mat result_mat = cv::Mat::zeros(src_img->height, src_img->width, CV_8UC1);
		//ofs1.open(filename);

		for(int y = 0 ; y < src_img->height ; y++){
			for(int x = 0 ; x < src_img->width ; x++){
				if(y == 0 || y == src_img->height-1 || x == 0 || x == src_img->width-1){
					src_img->imageData[y * src_img->widthStep + x] = (unsigned char)0;
				}
			}
		}

		// labeling processing
		LabelingBS labeling;
		int w = src_img->width;
		int h = src_img->height;

		unsigned char *src = new unsigned char [w * h];
		short *result = new short [w * h];

		for(int y = 0 ; y < h ; y++){
			for(int x = 0 ; x < w ; x++){
				src[y * w + x] = 
					(unsigned char)src_img->imageData[src_img->widthStep * y + x];
			}
		}

		/*cout << "labeling start" << endl;*/
		labeling.Exec(src, result, w, h, true, 5);

		int n = labeling.GetNumOfResultRegions();
		std::cout << "[" << k+1 << "] number of result regions: " << n << endl;

		for(int i = 0 ; i < n ; i++){
			IplImage *region_data = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
			cvZero(region_data);

			// making mask image
			for(int y = 0 ; y < h ; y++){
				for(int x = 0 ; x < w ; x++){
					if(result[y * w + x] == (i+1)){
						region_data->imageData[region_data->widthStep * y + x] = (unsigned char)255;
					}
				}
			}

			RegionInfoBS *ri = labeling.GetResultRegionInfo(i);
			CCFeature feature(ri, src_color, src_gray, region_data);

			double ccfeature[12];

			double *ContourFeature = feature.ContourFeature();
			double *SkeltonFeature = feature.SkeltonFeature();

			// Geometric Feature
			ccfeature[0] = feature.AreaRatio();
			ccfeature[1] = feature.AspectRatio();
			ccfeature[2] = feature.LengthRatio();

			//// Slelton Feature
			ccfeature[3] = SkeltonFeature[0];
			ccfeature[4] = SkeltonFeature[1];

			// Contour Feature
			ccfeature[5] = ContourFeature[0];
			ccfeature[6] = ContourFeature[1];
			ccfeature[7] = ContourFeature[2];
			ccfeature[8] = ContourFeature[3];

			// Shape Feature
			ccfeature[9] = feature.ContourRoughness();
			ccfeature[10] = feature.OccupyRatio();

			// Edge Feature
			ccfeature[11] = feature.SecondDerivativeOfContour();

			// Predict here
			cv::Mat text_sample = cv::Mat(1, VECTOR_DIM, CV_32F, ccfeature);
			int label = static_cast<int>(trees.predict(text_sample));
			if (label == 1) { // if predict as text
				for(int y = 0 ; y < h ; y++){
				    for(int x = 0 ; x < w ; x++){
					    if(region_data->imageData[region_data->widthStep * y + x]){
							result_mat.at<unsigned char>(y, x) = 255;
					    }
				    }
			    }
			}

			delete [] ContourFeature;
			delete [] SkeltonFeature;

			// デコンストラクタ
			feature.~CCFeature();

			// メモリ開放
			cvReleaseImage(&region_data);
		}

		delete [] src;
		delete [] result;


		// IplImage data release
		cvReleaseImage(&src_img);
		cvReleaseImage(&src_color);
		cvReleaseImage(&src_gray);
		cv::imwrite(output, result_mat);
	}

	
	return 0;
}