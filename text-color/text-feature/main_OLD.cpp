#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <ctime>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#pragma comment(lib, "opencv_core246.lib")
#pragma comment(lib, "opencv_imgproc246.lib")
#pragma comment(lib, "opencv_ml246.lib")
#pragma comment(lib, "opencv_highgui246.lib")

using namespace std;

#include "Labeling.h"
#include "ConnectedComp.h"
#include "GroundTruthReader.h"

bool GTCheck100(IplImage *gt_img, IplImage *mask_img);

#define MODE (1)
#define VECTOR_DIM (12)
#define WINDOW_RANGE (5)

typedef vector<double>::size_type vc_sz;

int main(){

	for(int k = 0 ; k < 150 ; k++){
		ofstream ofs1, ofs2;
		char input[512], input_gt[512], input_color[512], input_gray[512], output[512];
		char filename[512];

		IplImage *src_img, *gt_img, *dst_img;
		IplImage *src_color, *src_gray;
		IplImage *gt_check;

			// dataset2 file name set
			sprintf(input, "C:\\imagedata\\niblack\\dataset2_150_plus\\image%d.bmp", k+1);
			//sprintf(input_gt, "C:\\imagedata\\niblack\\dataset2_150_plus_GT\\image%d.bmp", k+1);
			sprintf(input_gt, "C:\\imagedata\\niblack\\dataset2_150_plus_GT_all_character\\image%d.bmp", k+1);
			sprintf(input_color, "C:\\imagedata\\dataset2_150\\image%d.jpg", k+1);
			sprintf(input_gray, "C:\\imagedata\\dataset2_150\\image%d.jpg", k+1);
			sprintf(filename, "C:\\imagedata\\textdataplus\\dataset2_150_cc12\\image%d.txt", k+1);
			//sprintf(filename, "C:\\imagedata\\textdataplus\\dataset2_150_allfeature\\image%d.txt", k+1);

		// IplImage data loading
		src_img = cvLoadImage(input, CV_LOAD_IMAGE_GRAYSCALE);
		src_color = cvLoadImage(input_color, CV_LOAD_IMAGE_ANYCOLOR);
		src_gray = cvLoadImage(input_gray, CV_LOAD_IMAGE_GRAYSCALE);
		gt_img = cvLoadImage(input_gt, CV_LOAD_IMAGE_GRAYSCALE);
		if(!src_img || !src_color || !gt_img){
			cout << "Not Found BMP Image Data!" << endl;
			continue;
		}
		dst_img = cvCloneImage(src_gray);
		cvZero(dst_img);
		gt_check = cvCloneImage(dst_img);
		ofs1.open(filename);

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
		cout << "[" << k+1 << "] number of result regions: " << n << endl;

		for(int i = 0 ; i < n ; i++){
			IplImage *region_data = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
			cvZero(region_data);

			IplImage *dilate_img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
			cvZero(dilate_img);

			// making mask image
			for(int y = 0 ; y < h ; y++){
				for(int x = 0 ; x < w ; x++){
					if(result[y * w + x] == (i+1)){
						region_data->imageData[region_data->widthStep * y + x] = (unsigned char)255;
					}
				}
			}

			// which is character of non-character?
			bool gt = GTCheck100(gt_img, region_data);
			if(gt){
				for(int y = 0 ; y < h ; y++){
					for(int x = 0 ; x < w ; x++){
						if(result[y * w + x] == (i+1)){
							gt_check->imageData[gt_check->widthStep * y + x] = (unsigned char)255;
						}
					}
				}
			}

			IplConvKernel *element = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_RECT, NULL);
			cvDilate(region_data, dilate_img, element, 10);

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


			// ファイル入出力
			if(gt){
				ofs1 << "1" << "\t";
			} else {
				ofs1 << "0" << "\t";
			}

			for(int j = 0 ; j < 12 ; j++){
				ofs1 << ccfeature[j] << "\t";
			}
			ofs1 << endl;
			/*ofs1 << size << "\t" << gx << "\t" << gy << "\t";
			for(int j = 0 ; j < 24 ; j++){
				ofs1 << mean(feature_vec[j]) << "\t";
				ofs1 << sd(feature_vec[j], true) << "\t";
				ofs1 << moment(feature_vec[j], 3, true) << "\t";
				ofs1 << moment(feature_vec[j], 4, true) << "\t";
			}
			ofs1 << endl;*/

			delete [] ContourFeature;
			delete [] SkeltonFeature;

			// デコンストラクタ
			feature.~CCFeature();

			// メモリ開放
			cvReleaseImage(&region_data);
			cvReleaseImage(&dilate_img);
		}

		/*char gt_file[512];
		sprintf(gt_file, "C:\\imagedata\\niblack\\icdar_train_fw_GroundTruth\\image%d.bmp", k+1);
		cvSaveImage(gt_file, gt_check);*/

		delete [] src;
		delete [] result;


		// IplImage data release
		cvReleaseImage(&src_img);
		cvReleaseImage(&src_color);
		cvReleaseImage(&src_gray);
		cvReleaseImage(&gt_img);
		cvReleaseImage(&dst_img);
		cvReleaseImage(&gt_check);
		ofs1.close();

	}
	
	cout << "\a";
	cout << "\a";
	
	return 0;
}

bool GTCheck100(IplImage *gt_img, IplImage *mask_img){
	int w = gt_img->width;
	int h = gt_img->height;

	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			int p = y * mask_img->widthStep + x;
			if(mask_img->imageData[p]){
				int d = gt_img->imageData[p];
				if(d < 105 && d > 95)
					return true;
			}
		}
	}

	return false;
}
