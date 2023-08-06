#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

#pragma comment(lib, "opencv_core245.lib")
#pragma comment(lib, "opencv_imgproc245.lib")
#pragma comment(lib, "opencv_ml245.lib")
#pragma comment(lib, "opencv_highgui245.lib")

// �v���g�^�C�v�֐�
int CountImageArea(IplImage *src_img);

// ��{�I�Ȋ֐���OpenCV�𗘗p���Ă��܂�
// �����t�H���W�[�ϊ��𗘗p���Ă���A
// �ڍׂ�OpenCV�̃��t�@�����X���Q�Ƃ��Ă�������

// ����������CCFeature�ɂ��Z�o����܂�

// �N���XCCFeature�𗘗p����ɂ́A
// ���x�����O�N���X���K�v(���x�����O�N���X�Ō�������΂���������j
// RegionInfoBS : ���x�����O���ʂ̏��ŁA
// �A�������̏c�����A�O�ڋ�`���̏�񂪊i�[����Ă���
// img : �J���[�摜(3ch BGR)
// mask : �A�������̈�(255�ˑΏۗ̈� 0�˔�Ώۗ̈�j
// img_gray : �O���C�X�P�[���摜(1ch gray)
// img_contours : �֊s���W ContourFeature�Z�o���ɕ`��


//HSVColorFeature �F����ǉ� 140628 �]��
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

	//double *HSVColorFeature(int H[256][256], int S[256][256], int V[256][256]);
	double *RGBColorFeature();

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

// �֊s������
// �o�� *f = new double [4]
// f[0] : CCHoles(�A�������̌��̐�)
// f[1] : AreaRatioS�iDilate������̖ʐ�/�A�������̖ʐρj
// f[2] : BoundaryS�iDilate������̋��E����/�A�������̋��E�����j
// f[3] : Compact�i�ʐ�/���E�����̓��j
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
		cvFindContours(src_img, storage1, &contours1, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	cvDrawContours(img_contours, contours1, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 0, 1);
	double Perimeter1 = cvArcLength(contours1);			// ���͒�
	double Area1 = (double) CountImageArea(src_img);	// �ʐ�
	
	// �f�[�^�m�ہ@(Dilation����)
	CvMemStorage *storage2 = cvCreateChildMemStorage(mem_contours);
	CvSeq *contours2 = NULL;

	// �֊s���o�@(dst_img : Dilation����)
	int find_contour_num2 =
		cvFindContours(dst_img, storage2, &contours2, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	double Perimeter2 = cvArcLength(contours2);			// ���͒��iDilation������
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
	f[0] = find_contour_num1 - 1; // Hole��,number-of-contours
	f[1] = Perimeter2 / Perimeter1;
	f[2] = Area2 / Area1;
	f[3] = Area1 / (Perimeter1 * Perimeter1);

	return f;
}

// Skelton�Ɋւ�������ʎZ�o
// f[0] : Skelton�̕���
// f[1] : Slelton�̕W���΍�
double *CCFeature::SkeltonFeature(){//scikit-image(skimage).morphology.skeletonize
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

	memset(d1, 0, w*h*sizeof(int));
	memset(d2, 0, w*h*sizeof(int));
	memset(d, 0, w*h*sizeof(int));
	memset(s, 0, w*h*sizeof(int));
	//std::fill_n(d1, w * h, 0);
	//std::fill_n(d2, w * h, 0);
	//std::fill_n(d, w * h, 0);
	//std::fill_n(s, w * h, 0);

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
// AreaRatio : �A�������̖ʐςƉ摜�T�C�Y�̔�
double CCFeature::AreaRatio(){
	double ratio;
	int w = mask->width;
	int h = mask->height;
	int area = ri->GetNumOfPixels();
	ratio = (double)area / (w * h);
	return ratio;
}

// LengthRatio : �A�������̊O�ڋ�`�̒��ӂƉ摜�̒��ӂ̔�
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

// AspectRatio : �A�������̒��ӂƒZ�ӂ̔�(����/�Z�Ӂj
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
// �A�������̋��E�̗��G��
// Opening�����O��̃T�C�Y�̕ω���
double CCFeature::ContourRoughness(){
	double ratio;
	IplImage *img_fill = cvCreateImage(cvSize(mask->width, mask->height), IPL_DEPTH_8U, 1);
	IplImage *img_opening = cvCreateImage(cvSize(mask->width, mask->height), IPL_DEPTH_8U, 1);
	IplImage *img_tmp = cvCloneImage(img_fill);

	// ���E��IplImage�ɕ`��
	cvDrawContours(img_fill, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), CV_FILLED, 1);

	// Opening����
	IplConvKernel *element = cvCreateStructuringElementEx (2, 2, 0, 0, CV_SHAPE_RECT, NULL);
	cvMorphologyEx(img_fill, img_opening, img_tmp, element, CV_MOP_OPEN, 1);

	// �ʐώZ�o�i��f�P�ʎZ�o�j
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

	// �f�[�^�J��
	cvReleaseImage(&img_fill);
	cvReleaseImage(&img_opening);
	cvReleaseImage(&img_tmp);
	cvReleaseStructuringElement(&element);

	return ratio;
}

// EdgeContrast : ���E���ɂ�����Q�������̐�Βl�̕���
double CCFeature::SecondDerivativeOfContour(){
	double value;

	IplImage *src_img = cvCloneImage(img_gray);
	IplImage *tmp_img = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_16S, 1);
	IplImage *laplace_img = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_8U, 1);
	IplImage *edgeline = cvCloneImage(img_contours);

	cvLaplace(src_img, tmp_img, 3);
	cvConvertScaleAbs(tmp_img, laplace_img);

	double sum = 0;

	int w = src_img->width;
	int h = src_img->height;
	int num = 0;
	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			if(edgeline->imageData[y * edgeline->widthStep + x] != 0){
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

// OccupyRatio : �A�������̃T�C�Y�ƊO�ڋ�`�̃T�C�Y�̔�
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
	cvConvertScaleAbs(tmp_img, sobel_img); //bytescale
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
			bool contour_edge_check = (img_contours->imageData[p] == 0);//scipy.misc.imfilter #2='find_edges'

			if(contour_edge_check){
				if(sobel_edge_check & canny_edge_check){
					count1++;
				} else {
					count2++;
				}
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




// HSV�F��Ԃɂ�������ƊO�̐F
//�}�X�N�̓����̋��E���O�Ƃ�������
//ex.�����Q�̎��C���͂P�C�O�͂Q���Y�������ɂȂ�
// f[0] : �F���̌�
// f[1] : �ʓx�̌�
// f[2] : ���x�̌�
//
double *CCFeature::RGBColorFeature(){
	double *f = new double [12];
	
	const int DIS = 2;//DISTANCE = 2;

	IplImage *src_img = cvCloneImage(mask);
	IplImage *dst_img = cvCloneImage(mask);
	cvZero(dst_img);

	// 4�ߖT�E�����ϊ�����
	int w = src_img->width;
	int h = src_img->height;
	int ws = src_img->widthStep;

	int *d = new int [w * h];

	std::fill_n(d, w * h, INT_MAX);

	// d:�����ϊ� distance transformation
	for(int y = DIS ; y <h-DIS  ; y++){
		for(int x = DIS ; x < w-DIS ; x++){
			//�}�X�N�����ǂ���
			if (src_img->imageData[y*ws + x]){
				//���E���ǂ���	
				if (!src_img->imageData[y*ws + (x-1)]
				 || !src_img->imageData[y*ws + (x+1)]
				 || !src_img->imageData[(y-1)*ws + x]
				 || !src_img->imageData[(y+1)*ws + x]){
					 //���E��f
					 d[y * w + x] = 0;
					 //�����P
					 if(d[y * w + (x-1)] > 1)	d[y * w + (x-1)]=1;
					 if(d[y * w + (x+1)] > 1)	d[y * w + (x+1)]=1;
					 if(d[(y-1) * w + x] > 1)	d[(y-1) * w + x]=1;
					 if(d[(y+1) * w + x] > 1)	d[(y+1) * w + x]=1;					 
					 //�����Q�ȏ�
					 for(int lev = 2; lev <= DIS; lev++){
						 //�����C����
						 if(d[y * w + (x-lev)] > lev)	d[y * w + (x-lev)]=lev;
						 if(d[y * w + (x+lev)] > lev)	d[y * w + (x+lev)]=lev;
						 if(d[(y-lev) * w + x] > lev)	d[(y-lev) * w + x]=lev;
						 if(d[(y+lev) * w + x] > lev)	d[(y+lev) * w + x]=lev;
						 //�΂�
						 for(int i =1; i<lev; i++){
							 int j = lev-i;
							 if(d[(y-j) * w + (x-i)] > lev)	d[(y-j) * w + (x-i)]=lev;
							 if(d[(y-j) * w + (x+i)] > lev)	d[(y-j) * w + (x+i)]=lev;
							 if(d[(y+j) * w + (x-i)] > lev)	d[(y+j) * w + (x-i)]=lev;
							 if(d[(y+j) * w + (x+i)] > lev)	d[(y+j) * w + (x+i)]=lev;
						 }//END.for�F�΂�
					 }//END.for�F�����Q�ȏ�
				 }//END.if�F���E
			}//�}�X�N��
		}
	}
	double in_R = 0.0;
	double in_G = 0.0;
	double in_B = 0.0;
	double out_R = 0.0;
	double out_G = 0.0;
	double out_B = 0.0;
	int RGBin_num = 0;
	int RGBout_num = 0;
	/**/
	for(int y = 0 ; y <h  ; y++){
		for(int x = 0 ; x < w ; x++){
			//�}�X�N�����ǂ���
			if (src_img->imageData[y*ws + x]){
				//����
				if(d[y * w + x] == DIS-1){
					in_R += (unsigned char)img->imageData[y*ws + x+0];
					in_G += (unsigned char)img->imageData[y*ws + x+1];
					in_B += (unsigned char)img->imageData[y*ws + x+2];
					RGBin_num++;
				 }
			}
			else{
				//�O��
				 if(d[y * w + x] == DIS){
					out_R += (unsigned char)img->imageData[y*ws + x+0];
					out_G += (unsigned char)img->imageData[y*ws + x+1];
					out_B += (unsigned char)img->imageData[y*ws + x+2];
					RGBout_num++;
				 }
			}
		}
	}
	/**/
	//�F���̕���
	double in_Hcos = 0;
	double in_Hsin = 0;
	double in_S = 0;
	double in_V = 0;
	int in_num =0;
	double out_Hcos = 0;
	double out_Hsin = 0;
	double out_S = 0;
	double out_V = 0;
	int out_num = 0;
	IplImage *hsv_img = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	cvCvtColor(img, hsv_img, CV_BGR2HSV);

	for(int y = 0 ; y <h  ; y++){
		for(int x = 0 ; x < w ; x++){
			//�}�X�N�����ǂ���
			if (src_img->imageData[y*ws + x]){
				//����
				if(d[y * w + x] == DIS-1){
					in_Hcos += cos(((double)hsv_img->imageData[y*ws + x]*2.0)*M_PI/180);
					in_Hsin += sin(((double)hsv_img->imageData[y*ws + x]*2.0)*M_PI/180);
					in_S += (unsigned char)hsv_img->imageData[y*ws + x+1];
					in_V += (unsigned char)hsv_img->imageData[y*ws + x+2];
					in_num++;
				 }
			}
			else{
				//�O��
				 if(d[y * w + x] == DIS){
					out_Hcos += cos(((double)hsv_img->imageData[y*ws + x]*2.0)*M_PI/180);
					out_Hsin += sin(((double)hsv_img->imageData[y*ws + x]*2.0)*M_PI/180);
					out_S += (unsigned char)hsv_img->imageData[y*ws + x+1];
					out_V += (unsigned char)hsv_img->imageData[y*ws + x+2];
					out_num++;
				 }
			}
		}
	}

	//HSV���ςƌ�
	int ave_inH = 0;
	int ave_inS = 0;
	int ave_inV = 0;
	int ave_outH = 0;
	int ave_outS = 0;
	int ave_outV = 0;
	
	int sumH = 0;
	int sumS = 0;
	int sumV = 0;
	if (in_num){
		ave_inH = floor((atan2(in_Hsin/(double)in_num, in_Hcos/(double)in_num)*90/M_PI + 90)+0.5);
		ave_inS = floor((in_S / (double)in_num)+0.5);
		ave_inV = floor((in_V / (double)in_num)+0.5);
		ave_outH = floor((atan2(out_Hsin/(double)out_num, out_Hcos/(double)out_num)*90/M_PI + 90)+0.5);
		ave_outS = floor((out_S / (double)out_num)+0.5);
		ave_outV = floor((out_V / (double)out_num)+0.5);
	f[0] = (int)(in_R/RGBin_num);
	f[1] = (int)(in_G/RGBin_num);
	f[2] = (int)(in_B/RGBin_num);
	f[3] = (int)(out_R/RGBout_num);
	f[4] = (int)(out_G/RGBout_num);
	f[5] = (int)(out_B/RGBout_num);
	f[6] = ave_inH;
	f[7] = ave_inS;
	f[8] = ave_inV;
	f[9] = ave_outH;
	f[10] = ave_outS;
	f[11] = ave_outV;

		//printf("%d,%d\n",in_num, out_num);
		//printf ("(%3d, %3d)(%3d, %3d)(%3d, %3d)\n",ave_inH, ave_outH,ave_inS, ave_outS,ave_inV, ave_outV);
		//assert(ave_inH >= 0 && ave_inS >= 0 && ave_inV >= 0 && ave_outH >= 0 && ave_outS >= 0 && ave_outV >= 0);
	
		//printf("H: %d, %d, %d\n", ave_inH, ave_outH, H[ave_inH][ave_outH]);
		//printf("S: %d, %d, %d\n", ave_inS, ave_outS, H[ave_inS][ave_outS]);
		//printf("V: %d, %d, %d\n", ave_inV, ave_outV, H[ave_inV][ave_outV]);
	/*
		int g[3][3]={18, 24, 6		//18 = 36/2
					,24, 16, 4
					,6, 4, 1};
		int pi_Hn;
		int mi_Hn;
		int po_Hn;
		int mo_Hn;

		int pi_Sn;
		int mi_Sn;
		int po_Sn;
		int mo_Sn;

		int pi_Vn;
		int mi_Vn;
		int po_Vn;
		int mo_Vn;
	
		/**
		for(int i = 0; i <180; i++){
			for(int j = 0; j <180; j++){
				printf("%d ", H[i][j]);
			}
			printf("\n");
		}/**

		for(int in = 0; in <= 2; in++){
			for(int out =0; out <= 2; out++){
				mi_Hn = ave_inH -in;
				pi_Hn = ave_inH + in;
				mo_Hn = ave_outH -out;
				po_Hn = ave_outH + out;
				if(mi_Hn < 0){
					mi_Hn += 180;
				}
				if(pi_Hn >= 180){
					pi_Hn -= 180;
				}
				if(mo_Hn < 0){
					mo_Hn += 180;
				}
				if(po_Hn >= 180){
					po_Hn -= 180;
				}
				//printf ("%d, %d, %.1f * %d\n", mi_n, mo_n, g[in][out], H[mi_n][mo_n]);
				if (!in || !out){
					//printf ("2*(%d, %d)* %.1f\n", in, out, g[in][out]);
					sumH +=  g[in][out]*H[mi_Hn][mo_Hn] + g[in][out]*H[pi_Hn][po_Hn];
				}else{
					//printf ("4*(%d, %d)* %.1f\n", in, out , g[in][out]);
					sumH += g[in][out]*H[mi_Hn][mo_Hn]
						  + g[in][out]*H[mi_Hn][po_Hn]
						  + g[in][out]*H[pi_Hn][mo_Hn]
						  + g[in][out]*H[pi_Hn][po_Hn];
				}
			
				mi_Sn = abs(ave_inS -in);
				pi_Sn = ave_inS + in;
				mo_Sn = abs(ave_outS -out);
				po_Sn = ave_outS + out;
				if(pi_Sn > 255)	pi_Sn = 510 - pi_Sn;//255 - (pi_Sn-255)
				if(po_Sn > 255)	po_Sn = 510 - po_Sn;//255 - (pi_Sn-255)
			
				//printf ("%d, %d, %d * %d\n", pi_Sn, po_Sn, g[in][out], S[pi_Sn][po_Sn]);
				if (!in || !out){
					sumS +=  g[in][out]*S[mi_Sn][mo_Sn] + g[in][out]*S[pi_Sn][po_Sn];
				}else{
					sumS += g[in][out]*S[mi_Sn][mo_Sn]
						  + g[in][out]*S[mi_Sn][po_Sn]
						  + g[in][out]*S[pi_Sn][mo_Sn]
						  + g[in][out]*S[pi_Sn][po_Sn];
				}
				mi_Vn = abs(ave_inV -in);
				pi_Vn = ave_inV + in;
				mo_Vn = abs(ave_outV -out);
				po_Vn = ave_outV + out;
				if(pi_Vn > 255)	pi_Vn = 510 - pi_Vn;//255 - (pi_Vn-255)
				if(po_Vn > 255)	po_Vn = 510 - po_Vn;//255 - (pi_Vn-255)
				if (!in || !out){
					sumV +=  g[in][out]*V[mi_Vn][mo_Vn] + g[in][out]*V[pi_Vn][po_Vn];
				}else{
					sumV += g[in][out]*V[mi_Vn][mo_Vn]
						  + g[in][out]*V[mi_Vn][po_Vn]
						  + g[in][out]*V[pi_Vn][mo_Vn]
						  + g[in][out]*V[pi_Vn][po_Vn];
				}

			}
		}
		/**/
	}
	else{
		return 0;
	}
	//printf ("(%d,%d):%d, %d %d %d %d\n",ave_inH, ave_outH, mi_Hn, mo_Hn, pi_Hn, po_Hn, H[ave_inH][ave_outH]);
	//printf ("(%d,%d) %d\n",ave_inV, ave_outV, V[ave_inV][ave_outV]);
	//printf ("%d, %d %d\n", sumH, sumS, sumV);
	
		cvReleaseImage(&hsv_img);
		/**/
		delete [] d;
		cvReleaseImage(&src_img);
		cvReleaseImage(&dst_img);
	//assert(sumH >= 0 && sumS >= 0 && sumV >= 0);

	return f;
}
