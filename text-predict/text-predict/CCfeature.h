#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>

// �v���g�^�C�v�֐�
int CountImageArea(IplImage *src_img);

class CCFeature {
private:
	RegionInfoBS *ri;		// ��{���
	IplImage *img;			// �J���[�摜
	IplImage *mask;			// �}�X�N�摜
	IplImage *img_gray;		// �O���C�X�P�[���摜
	IplImage *img_contours;	// �֊s���W

	CvSeq *contours;		
	CvMemStorage *mem_contours;

public:
	CCFeature(RegionInfoBS *riBS,
		IplImage *img_color, 
		IplImage *img_gray,
		IplImage *img_mask);

	double *ContourFeature();
	double *SkeltonFeature();
	double SecondDerivativeOfContour();

	double EdgeContrast();

	/*double *ContourFeature2();*/

	// Geometric Features
	double AreaRatio();
	double LengthRatio();
	double AspectRatio();

	// Shape Regularity Features
	double ContourRoughness();
	double OccupyRatio();

	int *BlockMaskArray();
	IplImage *BlockMaskImage();
	vector<int> BlockMaskIndex();

	IplImage *DilateImage(int iterations);

	~CCFeature();
};

// �R���X�g���N�^
CCFeature::CCFeature(RegionInfoBS *riBS,
					 IplImage *img_color,
					 IplImage *gray,
					 IplImage *img_mask){
	ri = riBS;
	img = img_color;
	mask = img_mask;
	img_gray = gray;
	img_contours = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	mem_contours = cvCreateMemStorage(0);
}

// �f�R���X�g���N�^
CCFeature::~CCFeature(){
	cvReleaseImage(&img_contours);
	cvReleaseMemStorage(&mem_contours);
}

double *CCFeature::ContourFeature(){
	double *f = new double [4];

	// �f�[�^�ǂݍ���
	IplImage *src_img = cvCloneImage(mask);		// ���摜
	IplImage *dst_img = cvCloneImage(mask);		// Dilation������
	cvZero(dst_img);

	// Dilation����
	IplConvKernel *element = cvCreateStructuringElementEx(5, 5, 2, 2, CV_SHAPE_RECT, NULL);
	cvDilate(src_img, dst_img, element, 1);

	// �f�[�^�m��
	CvMemStorage *storage1 = cvCreateChildMemStorage(mem_contours);
	CvSeq *contours1 = NULL;

	// �֊s���o�@�isrc_img�j
	int find_contour_num1 = 
		cvFindContours(src_img, storage1, &contours1, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cvDrawContours(img_contours, contours1, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 1, 1);
	double Perimeter1 = cvArcLength(contours1);			// ���͒�
	double Area1 = (double) CountImageArea(src_img);	// �ʐ�
	
	// �f�[�^�m�ہ@(Dilation����)
	CvMemStorage *storage2 = cvCreateChildMemStorage(mem_contours);
	CvSeq *contours2 = NULL;

	// �֊s���o�@(dst_img : Dilation����)
	int find_contour_num2 =
		cvFindContours(dst_img, storage2, &contours2, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	double Perimeter2 = cvArcLength(contours2);			// ���͒��iDilation������j
	double Area2 = (double) CountImageArea(dst_img);	// �ʐ� (Dilation������)

	contours = contours1;
	
	// �������J��
	cvReleaseStructuringElement(&element);
	cvReleaseImage(&src_img);
	cvReleaseImage(&dst_img);
	cvReleaseMemStorage(&storage1);
	cvReleaseMemStorage(&storage2);
	cvClearSeq(contours2);

	// �߂�l
	f[0] = find_contour_num1 - 1; // Hole��
	f[1] = Perimeter2 / Perimeter1;
	f[2] = Area2 / Area1;
	f[3] = Area1 / (Perimeter1 * Perimeter1);

	return f;
}

// Skelton�Ɋւ�������ʎZ�o
// f[0] : Skelton�̕���
// f[1] : Slelton�̕W���΍�
double *CCFeature::SkeltonFeature(){
	double *f = new double [2];

	IplImage *src_img = cvCloneImage(mask);
	IplImage *dst_img = cvCloneImage(mask);
	cvZero(dst_img);

	// 4�ߖT�E�����ϊ�����
	int w = src_img->width;
	int h = src_img->height;
	int ws = src_img->widthStep;

	int *d1 = new int [w * h];
	int *d2 = new int [w * h];
	int *d = new int [w * h];
	int *s = new int [w * h];

	std::fill_n(d1, w * h, 0);
	std::fill_n(d2, w * h, 0);
	std::fill_n(d, w * h, 0);
	std::fill_n(s, w * h, 0);

	// d1:
	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			if(src_img->imageData[y * ws + x]){
				d1[y * w + x] = INT_MAX;
			} else {
				d1[y * w + x] = 0;
			}
		}
	}

	// d2:
	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			if(x == 0 || x == w-1 || y == 0 || y == h-1){
				d2[y * w + x] = 0; 
				continue;
			}
			int tmp[3];
			tmp[0] = d1[y * w + x];
			tmp[1] = d2[(y-1) * w + x] + 1; 
			tmp[2] = d2[y * w + (x-1)] + 1;

			int min = INT_MAX;
			for(int i = 0 ; i < 3 ; i++){
				if(tmp[i] < min)
					min = tmp[i];
			}
			d2[y * w + x] = min;
		}
	}

	// d:�����ϊ� distance transformation
	for(int y = h-1 ; y >= 0 ; y--){
		for(int x = w-1 ; x >= 0 ; x--){
			if(x == 0 || x == w-1 || y == 0 || y == h-1){
				d[y * w + x] = 0; 
				continue;
			}
			int tmp[3];
			tmp[0] = d2[y * w + x];
			tmp[1] = d[(y+1) * w + x] + 1;
			tmp[2] = d[y * w + (x+1)] + 1;

			int min = INT_MAX;
			for(int i = 0 ; i < 3 ; i++){
				if(tmp[i] < min){
					min = tmp[i];
				}
			}
			d[y * w + x] = min;
		}
	}

	// s:���i(skelton)
	for(int y = 1 ; y < h-1 ; y++){
		for(int x = 1 ; x < w-1 ; x++){
			int tmp[4];
			tmp[0] = d[(y-1) * w + x];
			tmp[1] = d[(y+1) * w + x];
			tmp[2] = d[y * w + (x+1)];
			tmp[3] = d[y * w + (x-1)];

			int max = INT_MIN;
			for(int i = 0 ; i < 4 ; i++){
				if(max < tmp[i]){
					max = tmp[i];
				}
			}

			if(d[y * w + x] >= max){
				s[y * w + x] = d[y * w + x];
			} else {
				s[y * w + x] = 0;
			}
		}
	}

	// skelton�o��
	for(int y = 1 ; y < h-1 ; y++){
		for(int x = 1 ; x < w-1 ; x++){
			if(s[y * w + x])
			dst_img->imageData[y * ws + x] = 
				(unsigned char)(255);
		}
	}

	// �����ʎZ�o
	int sum = 0, sumsq = 0, counter = 0;
	for(int y = 1 ; y < h-1 ; y++){
		for(int x = 1 ; x < w-1 ; x++){
			if(s[y * w + x]){
				sum = sum + s[y * w + x];
				sumsq = sumsq + s[y * w + x] * s[y * w + x];
				counter++;
			}
		}
	}
	double stroke_mean = (double)sum / counter;
	double stroke_std = sqrt ( fabs ( (double)sumsq / counter) - stroke_mean * stroke_mean);

	// ���������
	delete [] d1;
	delete [] d2;
	delete [] d;
	delete [] s;
	cvReleaseImage(&src_img);
	cvReleaseImage(&dst_img);

	f[0] = stroke_mean;
	f[1] = stroke_std;

	return f;
}

// Geometric Feature
double CCFeature::AreaRatio(){
	double ratio;
	int w = mask->width;
	int h = mask->height;
	int area = ri->GetNumOfPixels();
	ratio = (double)area / (w * h);
	return ratio;
}

double CCFeature::LengthRatio(){
	double ratio;
	int w = mask->width;
	int h = mask->height;
	int size_x, size_y;
	ri->GetSize(size_x, size_y);
	int v1, v2;
	if(w > h) v1 = w;
	else v1 = h;
	if(size_x > size_y) v2 = size_x;
	else v2 = size_y;
	ratio = (double)v2 / v1;
	return ratio;
}

double CCFeature::AspectRatio(){
	double r1, r2;
	int size_x, size_y;
	ri->GetSize(size_x, size_y);
	r1 = (double)size_x / size_y;
	r2 = (double)size_y / size_x;
	if(r1 > r2) return r1;
	else return r2;
}

// Shape Regularity Feature
double CCFeature::ContourRoughness(){
	double ratio;
	IplImage *img_fill = cvCreateImage(cvSize(mask->width, mask->height), IPL_DEPTH_8U, 1);
	IplImage *img_opening = cvCreateImage(cvSize(mask->width, mask->height), IPL_DEPTH_8U, 1);
	IplImage *img_tmp = cvCloneImage(img_fill);
	cvDrawContours(img_fill, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), CV_FILLED, 1);
	IplConvKernel *element = cvCreateStructuringElementEx (3, 3, 1, 1, CV_SHAPE_RECT, NULL);
	cvMorphologyEx(img_fill, img_opening, img_tmp, element, CV_MOP_OPEN, 1);

	int area = ri->GetNumOfPixels();
	int area_open = 0;
	for(int y = 0 ; y < img_opening->height ; y++){
		for(int x = 0 ; x < img_opening->width ; x++){
			if(img_opening->imageData[y * img_opening->widthStep + x]){
				area_open++;
			}
		}
	}
	ratio = fabs((double)(area - area_open)) / (double)area;
	cvReleaseImage(&img_fill);
	cvReleaseImage(&img_opening);
	cvReleaseImage(&img_tmp);
	cvReleaseStructuringElement(&element);
	return ratio;
}


// �o�O�̉\���L
double CCFeature::SecondDerivativeOfContour(){
	double value;

	IplImage *src_img = cvCloneImage(img_gray);
	IplImage *tmp_img = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_16S, 1);
	IplImage *laplace_img = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_8U, 1);
	IplImage *edgeline = cvCloneImage(img_contours);

	float data[] = {0, -1, 0,
					-1, 4, -1,
					0, -1, 0};
	CvMat kernel = cvMat(3, 3, CV_32F, data);
	cvFilter2D(src_img, tmp_img, &kernel);
	cvConvertScaleAbs(tmp_img, laplace_img);

	//cvLaplace(src_img, tmp_img, 3);
	

	double sum = 0;

	int w = src_img->width;
	int h = src_img->height;
	int num = 0;
	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			if(edgeline->imageData[y * edgeline->widthStep + x] != (unsigned char)0){
				sum = sum + 
					(unsigned char)laplace_img->imageData[y * laplace_img->widthStep + x];
				num++;
			}
		}
	}
	value = sum / num;

	cvReleaseImage(&src_img);
	cvReleaseImage(&tmp_img);
	cvReleaseImage(&laplace_img);
	cvReleaseImage(&edgeline);
	return value;
}

double CCFeature::OccupyRatio(){
	double ratio;
	int area = ri->GetNumOfPixels();
	int size_x, size_y;
	ri->GetSize(size_x, size_y);
	int bbarea = size_x * size_y;
	ratio = (double)area / bbarea;
	return ratio;
}



// �G�b�W�Ɋւ�������ʎZ�o
// �֊s������ɍs���K�v������
double CCFeature::EdgeContrast(){
	double f;

	// �������m��
	IplImage *src_img = cvCloneImage(img_gray);
	IplImage *tmp_img = cvCreateImage(cvGetSize(img_gray), IPL_DEPTH_16S, 1);
	IplImage *sobel_img = cvCreateImage(cvGetSize(img_gray), IPL_DEPTH_8U, 1);
	IplImage *canny_img = cvCreateImage(cvGetSize(img_gray), IPL_DEPTH_8U, 1);

	// �G�b�W�摜�̍쐬
	cvSobel(src_img, tmp_img, 1, 1);
	cvConvertScaleAbs(tmp_img, sobel_img);
	cvCanny(src_img, canny_img, 50.0, 200.0);

	// canny & sobel�̃G�b�W����
	int count1 = 0, count2 = 0;
	int w = src_img->width;
	int h = src_img->height;
	int ws = src_img->widthStep;

	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			int p = y * ws + x;
			bool sobel_edge_check = (sobel_img->imageData[p] == 0);
			bool canny_edge_check = (canny_img->imageData[p] == 0);
			bool contour_edge_check = (img_contours->imageData[p] == 0);

			if(contour_edge_check){
				if(sobel_edge_check || canny_edge_check){
					count1++;
				}
				count2++;
			}
		}
	}

	// �������J��
	cvReleaseImage(&src_img);
	cvReleaseImage(&tmp_img);
	cvReleaseImage(&sobel_img);
	cvReleaseImage(&canny_img);
	f = (double)count1 / count2;
	return f;
}

int *CCFeature::BlockMaskArray(){
	IplImage *src_img = cvCloneImage(mask);
	const int block_size = 16;
	int bw = src_img->width / block_size;
	int bh = src_img->height / block_size;

	int *block_region = new int [bw * bh];
	for(int i = 0 ; i < bw * bh ; i++) block_region[i] = 0;

	for(int y = 0 ; y < src_img->height ; y++){
		for(int x = 0 ; x < src_img->width ; x++){
			if(src_img->imageData[y * src_img->widthStep + x] != static_cast<uchar>(0)){
				int nx = x / block_size;
				int ny = y / block_size;
				block_region[ny * bw + nx] = 1;
			}
		}
	}
	cvReleaseImage(&src_img);
	return block_region;
}

IplImage *CCFeature::BlockMaskImage(){
	IplImage *src_img = cvCloneImage(mask);
	IplImage *dst_img = cvCloneImage(mask);
	cvZero(dst_img);

	const int block_size = 16;
	int bw = src_img->width / block_size;
	int bh = src_img->height / block_size;

	int *block_region = new int [bw * bh];
	for(int i = 0 ; i < bw * bh ; i++) block_region[i] = 0;

	for(int y = 0 ; y < src_img->height ; y++){
		for(int x = 0 ; x < src_img->width ; x++){
			if(src_img->imageData[y * src_img->widthStep + x] != static_cast<uchar>(0)){
				int nx = x / block_size;
				int ny = y / block_size;
				block_region[ny * bw + nx] = 1;
			}
		}
	}
	for(int by = 0 ; by < bh ; by++){
		for(int bx = 0 ; bx < bw ; bx++){
			if(block_region[by * bw + bx] == 1){
				for(int y = by * block_size ; y < (by+1) * block_size ; y++){
					for(int x = bx * block_size ; x < (bx+1) * block_size ; x++){
						dst_img->imageData[y * dst_img->widthStep + x] = static_cast<uchar>(255);
					}
				}
			}
		}
	}
	delete [] block_region;
	cvReleaseImage(&src_img);
	return dst_img;
}

vector<int> CCFeature::BlockMaskIndex(){
	vector<int> index;
	IplImage *src_img = cvCloneImage(mask);
	const int block_dis = 16;
	int bw = src_img->width / block_dis;
	int bh = src_img->height / block_dis;

	int *block_region = new int [bw * bh];
	for(int i = 0 ; i < bw * bh ; i++) block_region[i] = 0;

	for(int y = 0 ; y < src_img->height ; y++){
		for(int x = 0 ; x < src_img->width ; x++){
			if(src_img->imageData[y * src_img->widthStep + x] != static_cast<uchar>(0)){
				int nx = x / block_dis;
				int ny = y / block_dis;
				//block_region[ny * bw + nx] = 1;
				index.push_back(ny * bw + nx);
			}
		}
	}
	cvReleaseImage(&src_img);
	return index;
	
}

IplImage *CCFeature::DilateImage(int iterations){
	IplImage *src_img = cvCloneImage(mask);
	IplImage *dst_img = cvCloneImage(mask);
	IplConvKernel *element = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_RECT, NULL);
	cvDilate(src_img, dst_img, element, iterations);
	cvReleaseStructuringElement(&element);
	cvReleaseImage(&src_img);
	return dst_img;
}


// �O���C�X�P�[���摜�ɂ����āA�P�x�l��0�łȂ��̈�̉�f�����J�E���g���Ė߂�l��
int CountImageArea(IplImage *src_img){
	int c = 0;
	int w = src_img->width;
	int h = src_img->height;
	int ws = src_img->widthStep;

	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			if(src_img->imageData[y * ws + x]){
				c++;
			}
		}
	}
	return c;
}

IplImage *MaskImage(IplImage *img, IplImage *mask){
	int ch = img->nChannels;
	IplImage *dst_img = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, ch);
	cvZero(dst_img);
	cvCopy(img, dst_img, mask);
	return dst_img;
}

bool GTCheck(IplImage *gt_img, int x, int y){
	int p = y * gt_img->widthStep + 3 * x;
	unsigned char point[3];
	point[0] = gt_img->imageData[p + 0];
	point[1] = gt_img->imageData[p + 1];
	point[2] = gt_img->imageData[p + 2];
	bool check = (point[0] < (unsigned char)10 &&
		point[1] < (unsigned char)10 &&
		point[2] > (unsigned char)250);
	return check;
}

bool GTCheck2(IplImage *gt_img, IplImage *mask_img){
	unsigned char point[3];
	int w = gt_img->width;
	int h = gt_img->height;

	/*IplImage *erode_img = cvCloneImage(mask_img);
	IplConvKernel *element = cvCreateStructuringElementEx (9, 9, 4, 4, CV_SHAPE_RECT, NULL);
	cvErode(mask_img, erode_img, element, 1);*/

	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){

			int p1 = y * mask_img->widthStep + x;
			int p3 = y * gt_img->widthStep + 3 * x;

			if(mask_img->imageData[p1]){
				point[0] = gt_img->imageData[p3 + 0];
				point[1] = gt_img->imageData[p3 + 1];
				point[2] = gt_img->imageData[p3 + 2];

				if(point[0] < (unsigned char)10 &&
				point[1] < (unsigned char)10 &&
				point[2] > (unsigned char)250){
					return true;
				}
			}

		}
	}

	/*cvReleaseImage(&erode_img);*/

	return false;
}


//double CCFeature::Compact(){
//	double ratio;
//	int area = ri->GetNumOfPixels();
//	double perimeter = cvArcLength(contours);
//	ratio = (double)area / (perimeter * perimeter);
//	return ratio;
//}

 //�֊s�����Ɋւ�������ʎZ�o
 //f[0] : �z�[����
 //f[1] : ���͒��� (Dilation�O��)
 //f[2] : �ʐϔ� (Dilation�O��)
 //f[3] : �ʐρE���͒���
//double *CCFeature::ContourFeature(){
//	double *f = new double [4];
//
//	// �f�[�^�ǂݍ���
//	IplImage *src_img = cvCloneImage(mask);		// ���摜
//	IplImage *dst_img = cvCloneImage(mask);		// Dilation������
//	cvZero(dst_img);
//
//	// Dilation����
//	IplConvKernel *element = cvCreateStructuringElementEx(5, 5, 2, 2, CV_SHAPE_RECT, NULL);
//	cvDilate(src_img, dst_img, element, 1);
//
//	// �f�[�^�m��
//	CvMemStorage *storage1 = cvCreateChildMemStorage(mem_contours);
//	CvSeq *contours1 = NULL;
//	// �֊s���o�@�isrc_img�j
//	int find_contour_num1 = 
//		cvFindContours(src_img, storage1, &contours1, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
//	double Perimeter1 = cvArcLength(contours1);			// ���͒�
//	double Area1 = (double) CountImageArea(src_img);	// �ʐ�
//	*cvDrawContours(img_contours, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 0, 1);*/
//	
//	// �f�[�^�m�ہ@(Dilation����)
//	//CvMemStorage *storage2 = cvCreateChildMemStorage(mem_contours);
//	//CvSeq *contours2 = NULL;
//
//	//// �֊s���o�@(dst_img : Dilation����)
//	//int find_contour_num2 =
//	//	cvFindContours(dst_img, storage2, &contours2, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
//	//double Perimeter2 = cvArcLength(contours2);			// ���͒��iDilation������j
//	//double Area2 = (double) CountImageArea(dst_img);	// �ʐ� (Dilation������)
//
//	//CvMemStorage *storage = cvCreateChildMemStorage(mem_contours);
//	///*CvSeq *contours1 = NULL;*/
//	//CvSeq *contours = NULL;
//	//// �֊s���o�@�isrc_img�j
//	///*int find_contour_num1 = 
//	//	cvFindContours(src_img, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_NONE);*/
//	//int find_contour_num1 = 
//	//	cvFindContours(src_img, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
//	//double Perimeter1 = cvArcLength(contours);			// ���͒�
//	//double Area1 = (double) CountImageArea(src_img);	// �ʐ�
//	///*cvDrawContours(img_contours, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 0, 1);*/
//
//	//cvReleaseMemStorage(&storage);
//	//
//	//// �f�[�^�m�ہ@(Dilation����)
//	//storage = cvCreateChildMemStorage(mem_contours);
//	//cvClearSeq(contours);
//	/*CvSeq *contours2 = NULL;*/
//
//	// �֊s���o�@(dst_img : Dilation����)
//	int find_contour_num2 =
//		cvFindContours(dst_img, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
//	double Perimeter2 = cvArcLength(contours);			// ���͒��iDilation������j
//	double Area2 = (double) CountImageArea(dst_img);	// �ʐ� (Dilation������)
//
//	cvReleaseMemStorage(&storage);
//	
//	// �������J��
//	cvReleaseStructuringElement(&element);
//	cvReleaseImage(&src_img);
//	cvReleaseImage(&dst_img);
//	/*cvReleaseMemStorage(&storage1);
//	cvReleaseMemStorage(&storage2);*/
//	/*cvClearSeq(contours1);
//	cvClearSeq(contours2);*/
//
//	// �߂�l
//	f[0] = find_contour_num1 - 1; // Hole��
//	f[1] = Perimeter2 / Perimeter1;
//	f[2] = Area2 / Area1;
//	f[3] = Area1 / (Perimeter1 * Perimeter1);
//
//	//f[0] = 1; // Hole��
//	//f[1] = 0.5;
//	//f[2] = 0.5;
//	//f[3] = 0.5;
//
//
//	return f;
//}