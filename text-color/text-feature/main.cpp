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

#pragma comment(lib, "opencv_core245.lib")
#pragma comment(lib, "opencv_imgproc245.lib")
#pragma comment(lib, "opencv_ml245.lib")
#pragma comment(lib, "opencv_highgui245.lib")

using namespace std;

#include "Labeling.h"
#include "ConnectedComp.h"
#include "GroundTruthReader.h"


bool GTCheck100(const cv::Mat &img, IplImage *mask_img);

#define MODE (1)
//#define VECTOR_DIM (12)	+color
#define VECTOR_DIM (12)
#define WINDOW_RANGE (5)

typedef vector<double>::size_type vc_sz;

#define TrainingNum 3018//2000

void ReadCountFile(FILE *f, int out[256][256]);

//HSVColorFeature 色情報を追加 140628 江口
int main(){
	/**
	//HSV表の読み込み
	FILE *f_CountH;
	FILE *f_CountS;
	FILE *f_CountV;
	int CountH[256][256];
	int CountS[256][256];
	int CountV[256][256];
	f_CountH = fopen("C:/Image/exp/01/CountColorH.txt", "r");
	f_CountS = fopen("C:/Image/exp/01/CountColorS.txt", "r");
	f_CountV = fopen("C:/Image/exp/01/CountColorV.txt", "r");
	if( f_CountH == NULL || f_CountS == NULL || f_CountV == NULL){
		printf("ファイルが読み込めません\n");
		exit(-1);
	}
	ReadCountFile(f_CountH, CountH);
	ReadCountFile(f_CountS, CountS);
	ReadCountFile(f_CountV, CountV);
	/*
	for(int i = 0; i <180; i++){
		for(int j = 0; j <180; j++){
			printf("%d ", CountH[i][j]);
		}
		printf("\n");
	}*/
	/**
	fclose(f_CountH);
	fclose(f_CountS);
	fclose(f_CountV);
	FILE *f_dataset;
	f_dataset = fopen("C:/Image/exp/01/TrainingData.txt", "r");
	if(f_dataset == NULL){
		printf("ファイルが読み込めません\n");
		exit(-1);
	}
	char row[10000];
	char *tok;
	int dataset[TrainingNum];

	while( fgets( row, 10000, f_dataset ) != NULL ){
		tok = strtok(row, " ");
		for(int rn = 0; rn < TrainingNum; rn++){
			dataset[rn] = atoi(tok);
			tok = strtok(NULL, " ");
		}
	}
	//for(int dn=0; dn < 2000;dn++)
	//	printf("%d\n",dataset[dn]);
	/**/
	/////////////////////////////////追加
	int margin = 0;
	for(int Tn = 0 ; Tn < TrainingNum-margin; Tn++){
		int k = Tn+margin+1;//dataset[Tn];
		ofstream ofs1, ofs2;
		char input[512], input_gt[512], input_color[512], input_gray[512], output[512];
		char filename[512];

		IplImage *src_img;
		IplImage *src_color, *src_gray;
		//IplImage *gt_check;
		//ground truth directory
		// To directory where NIBLACK results exist
		sprintf(input, "C:/Image/Niblack/FIS_%d.bmp", k);
		// To ground truth directory
	    sprintf(input_gt, "C:/Image/FIS/FIS_%d_GT.bmp", k);
		// To original color image
		sprintf(input_color, "C:/Image/FIS/FIS_%d.jpg", k);
		// To original color image, but read as gray image
		sprintf(input_gray, "C:/Image/FIS/FIS_%d.jpg", k);
		// To directory where feature file will be saved
		sprintf(filename,"C:/Image/test2/color/FIS_%d.txt", k);

		// IplImage data loading
		src_img = cvLoadImage(input, CV_LOAD_IMAGE_GRAYSCALE);
		src_color = cvLoadImage(input_color, CV_LOAD_IMAGE_ANYCOLOR);
		src_gray = cvLoadImage(input_gray, CV_LOAD_IMAGE_GRAYSCALE);
		//std::cout << "GT:" << input_gt << std::endl;
		IplImage * gt = cvLoadImage(input_gt);
		cv::Mat gt_img(gt);
		if(!src_img || !src_color || gt_img.empty()){
			cout << "Not Found BMP Image Data! " << k << endl;
			continue;
		}
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
		
		clock_t start, end;
		start = clock();

		/*cout << "labeling start" << endl;*/
		labeling.Exec(src, result, w, h, true, 5);

		int n = labeling.GetNumOfResultRegions();
		cout << "[" << k << "] number of result regions: " << n << endl;

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

			// which is character of non-character?
			bool gt = GTCheck100(gt_img, region_data);
			/*if(gt){
				for(int y = 0 ; y < h ; y++){
					for(int x = 0 ; x < w ; x++){
						if(result[y * w + x] == (i+1)){
							gt_check->imageData[gt_check->widthStep * y + x] = (unsigned char)255;
						}
					}
				}
			}*/
			//std::cout << "text ?" << gt << std::endl;
			//cvShowImage("mask", region_data);
			//cvWaitKey(0);

			RegionInfoBS *ri = labeling.GetResultRegionInfo(i);
			//+color
			CCFeature feature(ri, src_color, src_gray, region_data);
			/**
			//double ccfeature[12];		+color
			double ccfeature[15];

			double *ContourFeature = feature.ContourFeature();
			double *SkeltonFeature = feature.SkeltonFeature();
			//+color
			//double *HSVColorFeature = feature.HSVColorFeature(CountH, CountS, CountV);

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
			/**
			//+ HSV Color Feature 
			ccfeature[12] = HSVColorFeature[0];
			ccfeature[13] = HSVColorFeature[1];
			ccfeature[14] = HSVColorFeature[2];
			/**/
			double ccfeature[VECTOR_DIM];
			double *RGBColorFeature = feature.RGBColorFeature();
			if (RGBColorFeature == 0){
				delete[] RGBColorFeature;
			// デコンストラクタ
				feature.~CCFeature();

			// メモリ開放
				cvReleaseImage(&region_data);
				break;

			}
			ccfeature[0] = RGBColorFeature[0];
			ccfeature[1] = RGBColorFeature[1];
			ccfeature[2] = RGBColorFeature[2];
			ccfeature[3] = RGBColorFeature[3];
			ccfeature[4] = RGBColorFeature[4];
			ccfeature[5] = RGBColorFeature[5];
			ccfeature[6] = RGBColorFeature[6];
			ccfeature[7] = RGBColorFeature[7];
			ccfeature[8] = RGBColorFeature[8];
			ccfeature[9] = RGBColorFeature[9];
			ccfeature[10] = RGBColorFeature[10];
			ccfeature[11] = RGBColorFeature[11];

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

			//delete [] ContourFeature;
			//delete [] SkeltonFeature;
			//+color
			delete[]  RGBColorFeature;

			// デコンストラクタ
			feature.~CCFeature();

			// メモリ開放
			cvReleaseImage(&region_data);
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
		//cvReleaseImage(&gt_check);
		ofs1.close();

	}
	
	cout << "\a";
	cout << "\a";
	
	return 0;
}

bool GTCheck100(const cv::Mat &gt_img, IplImage *mask_img){
	int w = gt_img.cols;
	int h = gt_img.rows;
	assert (gt_img.channels() == 3);
	int total = cvCountNonZero(mask_img);
	int count = 0;
	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			int p = y * mask_img->widthStep + x;
			if(mask_img->imageData[p]){
				cv::Vec3b pixel = gt_img.at<cv::Vec3b>(y,x);
				//std::cout << d0 << " " << d1 << " " << d2 << std::endl;
				if(pixel[0] != 255 || pixel[1] != 255 || pixel[2] != 255)
					++ count;
			}
		}
	}
	//cvShowImage("mask", mask_img);
	//cvWaitKey(0);

	return (static_cast<float>(count) / total > 0.5);
}

void ReadCountFile(FILE *f, int out[256][256]){
	char row[50];
	char *tok;
	unsigned char i , j;

	while( fgets( row, 50, f ) != NULL ){
		if(!(row[0] == '#' || row[0] == '\r' || row[0] == '\n')){
			//printf("%s",row);
			tok = strtok(row, " ");
			i = (unsigned char)atoi(tok);
			tok = strtok(NULL, " ");
			j = (unsigned char)atoi(tok);
			tok = strtok(NULL, " ");
			out[i][j] = (int)atoi(tok);

		}
	}
}