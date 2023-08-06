#pragma once

#include <iostream>
#include <fstream>
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

#define CCFEATURE_NUM (12)

int main(){

	clock_t start, end;

	vector <float> cc_vec;
	vector <int> label_vec, texton_label_vec, dilate_label_vec;

	for(int k = 0 ; k < 150 ; k++){
		ifstream ifs1;
		char ccfile[512];
		// To feature file
		sprintf(ccfile, "C:\\imagedata\\textdata\\dataset1_150_cc12\\image%d.txt", k+1);
		ifs1.open(ccfile);

		int load_label;
		float load_data1[CCFEATURE_NUM];
		
		while(!ifs1.eof()){
			ifs1 >> load_label;
			for(int i = 0 ; i < CCFEATURE_NUM ; i++){
				if(ifs1.eof()){
					goto LOOPEND;
				}
				ifs1 >> load_data1[i];
			}
		}
LOOPEND:

		ifs1.close();
	} 

	cout << "data load finish" << "\a" << endl;

	int sample_num = label_vec.size();

	CvMat data_mat1;
	CvMat res_mat1;

	cvInitMatHeader(&data_mat1, sample_num, CCFEATURE_NUM, CV_32F, &cc_vec[0]);
	cvInitMatHeader(&res_mat1, sample_num, 1, CV_32S, &label_vec[0]);

	
	for(int i = 0 ; i < 10 ; i++){
		cout << res_mat1.data.i[i] << "\t";
		for(int j = 0 ; j < CCFEATURE_NUM ; j++){
			cout << data_mat1.data.fl[i * CCFEATURE_NUM + j] << "\t";
		}
		cout << endl;
	}

	float priors[2];
	priors[0] = 0.90f;
	priors[1] = 0.10f;

	start = clock();

	CvRTrees forest1 = CvRTrees();
	CvMat *var_type = cvCreateMat(data_mat1.cols+1, 1, CV_8U);
	cvSet( var_type, cvScalarAll(CV_VAR_NUMERICAL) );
	cvSetReal1D( var_type, data_mat1.cols, CV_VAR_CATEGORICAL );
	forest1.train(&data_mat1, CV_ROW_SAMPLE, &res_mat1, 0, 0, var_type, 0,
		CvRTParams(10, 10, 0, false, 2, priors, true, 4, 500, 0.01f, CV_TERMCRIT_ITER));
	// To directory classifier will be saved
	forest1.save("C:\\imagedata\\textdata\\ml\\rf\\cc12_10_4_500_0.90_75-150.xml");
	//forest1.save("C:\\imagedata\\textdata\\ml\\rf\\cc12_10_4_500_0.90_0-75.xml");

	end = clock();
	cout << (double)(end - start) / CLOCKS_PER_SEC << " second" << endl;

	return 0;
}