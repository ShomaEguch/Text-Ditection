
// LABELDATA ENUM
enum CONTEXT_LABEL { NOTHING, BUILDING, GRASS, TREE, COW, HORSE, SHEEP, SKY, 
					MOUNTAIN, AEROPLANE, WATER, FACE, CAR, BICYCLE, FLOWER, SIGN,
					BIRD, BOOK, CHAIR, ROAD, CAT, DOG, BODY, BOAT};

// LABELING COLOR
const unsigned char label_nothing[3] = {0, 0, 0};
const unsigned char label_building[3] = {128, 0, 0};
const unsigned char label_grass[3] = {0, 128, 0};
const unsigned char label_tree[3] = {128, 128, 0};
const unsigned char label_cow[3] = {0, 0, 128};

const unsigned char label_horse[3] = {128, 0, 128};
const unsigned char label_sheep[3] = {0, 128, 128};
const unsigned char label_sky[3] = {128, 128, 128};
const unsigned char label_mountain[3] = {64, 0, 0};
const unsigned char label_aeroplane[3] = {192, 0, 0};

const unsigned char label_water[3] = {64, 128, 0};
const unsigned char label_face[3] = {192, 128, 0};
const unsigned char label_car[3] = {64, 0, 128};
const unsigned char label_bicycle[3] = {192, 0, 128};
const unsigned char label_flower[3] = {64, 128, 128};

const unsigned char label_sign[3] = {192, 128, 128};
const unsigned char label_bird[3] = {0, 64, 0};
const unsigned char label_book[3] = {128, 64, 0};
const unsigned char label_chair[3] = {0, 192, 0};
const unsigned char label_road[3] = {128, 64, 128};

const unsigned char label_cat[3] = {0, 192, 128};
const unsigned char label_dog[3] = {128, 192, 128};
const unsigned char label_body[3] = {64, 64, 0};
const unsigned char label_boat[3] = {192, 64, 0};



int label_num(const unsigned char *label){
	int r = static_cast<int>(label[0]);
	int g = static_cast<int>(label[1]);
	int b = static_cast<int>(label[2]);
	return ( r * ( 256 * 256 ) + g * ( 256 ) + b );
}

#define num_nothing label_num(label_nothing);

// gt_img : Ground Truth image
// x, y : point coordinate
// gt_color : color array = { r, g, b }
bool GroundTruthCheck(const IplImage *gt_img, int x, int y, const unsigned char *gt_color){
	int p = y * gt_img->widthStep + x * gt_img->nChannels;
	unsigned char b, g, r;
	b = static_cast<unsigned char>(gt_img->imageData[p]);
	g = static_cast<unsigned char>(gt_img->imageData[p+1]);
	r = static_cast<unsigned char>(gt_img->imageData[p+2]);

	if(r == gt_color[0] && g == gt_color[1] && b == gt_color[2]){
		return true;
	} else {
		return false;
	}
}

// Microsoftの領域分割データベース（環境認識用）の
// 学習用画像から画素毎に付与ラベルを読み取る
int GroundTruthLabel(const IplImage *gt_img, int x, int y){
	int p = y * gt_img->widthStep + x * gt_img->nChannels;
	int b, g, r;
	b = static_cast<unsigned char>(gt_img->imageData[p]);
	g = static_cast<unsigned char>(gt_img->imageData[p+1]);
	r = static_cast<unsigned char>(gt_img->imageData[p+2]);

	int check_num = r * ( 256 * 256 ) + g * ( 256 ) + b;
	int label = INT_MAX;

	switch(check_num){
		case 0:			// 0 0 0
			label = NOTHING;
			break;
		case 128 * 256 * 256:	// 128 0 0
			label = BUILDING;
			break;
		case 128 * 256:			// 0 128 0
			label = GRASS;
			break;
		case 128 * 256 * 256 + 128 * 256:		// 128 128 0
			label = TREE;
			break;
		case 128:			// 0 0 128
			label = COW;
			break;

		case 128 * 256 * 256 + 128:		// 128 0 128
			label = HORSE;
			break;
		case 128 * 256 + 128:			// 0 128 128
			label = SHEEP;
			break;
		case 128 * 256 * 256 + 128 * 256 + 128:			// 128 128 128
			label = SKY;
			break;
		case 64 * 256 * 256:	// 64 0 0
			label = MOUNTAIN;
			break;
		case 192 * 256 * 256:			// 192 0 0
			label = AEROPLANE;
			break;

		case 64 * 256 * 256 + 128 * 256:			// 64 128 0
			label = WATER;
			break;
		case 192 * 256 * 256 + 128 * 256:			// 192 128 0
			label = FACE;
			break;
		case 64 * 256 * 256 + 128:			// 64 0 128
			label = CAR;
			break;
		case 192 * 256 * 256 + 128:			// 192 0 128
			label = BICYCLE;
			break;
		case 64 * 256 * 256 + 128 * 256 + 128:			// 64 128 128
			label = FLOWER;
			break;

		case 192 * 256 * 256 + 128 * 256 + 128:			// 192 128 128
			label = SIGN;
			break;
		case 64 * 256:			// 0 64 0
			label = BIRD;
			break;
		case 128 * 256 * 256 + 64 * 256:			// 128 64 0
			label = BOOK;
			break;
		case 192 * 256:			// 0 192 0
			label = CHAIR;
			break;
		case 128 * 256 * 256 + 64 * 256 + 128:		// 128 64 128
			label = ROAD;
			break;

		case 192 * 256 + 128:			// 0 192 128
			label = CAT;
			break;
		case 128 * 256 * 256 + 192 * 256 + 128:				// 128 192 128
			label = DOG;
			break;
		case 64 * 256 * 256 + 64 * 256:			// 64 64 0
			label = BODY;
			break;
		case 192 * 256 * 256 + 64 * 256:			// 192 64 0
			label = BOAT;
			break;
		default:
			label = -1;
			break;
	}
	return label;
}

const unsigned char labelcolor[24][3] = {

{0, 0, 0},
{128, 0, 0},
{0, 128, 0},
{128, 128, 0},
{0, 0, 128},

{128, 0, 128},
{0, 128, 128},
{128, 128, 128},
{64, 0, 0},
{192, 0, 0},

{64, 128, 0},
{192, 128, 0},
{64, 0, 128},
{192, 0, 128},
{64, 128, 128},

{192, 128, 128},
{0, 64, 0},
{128, 64, 0},
{0, 192, 0},
{128, 64, 128},

{0, 192, 128},
{128, 192, 128},
{64, 64, 0},
{192, 64, 0}

};

// オリジナルデータセットのラベル読み取り用関数
// BMP画像ではなくJPEG画像で作成したため
// 各ラベルの輝度値に幅があります
// 設定クラスは
// 空・草木・看板・地面・建造物・ポール
int dataset150_label(IplImage *gt, int x, int y){
	int ch = gt->nChannels;
	if(ch != 3){
		cerr << "Ground Truth Image channel Error (3ch only)" << endl;
		return -1;
	}
	int p = y * gt->widthStep + x * gt->nChannels;
	int b = static_cast<unsigned char>(gt->imageData[p]);
	int g = static_cast<unsigned char>(gt->imageData[p+1]);
	int r = static_cast<unsigned char>(gt->imageData[p+2]);

	// SKY : RGB -> 0 0 255
	if(r < 10 && g < 10 && b > 245)
		return 1;	

	// TREE&GRASS : RGB -> 0 255 0
	if(r < 10 && g > 245 && b < 10)	
		return 2;	

	// SIGN : RGB -> 255 255 0
	if(r > 245 && g > 245 && b < 10)	
		return 3;	

	// GROUND : RGB -> 255 0 0
	if(r > 245 && g < 10 && b < 10)	
		return 4;	

	// GROUND(GRASS) : RGB -> 255 0 255
	if(r > 245 && g < 10 && b > 245)	
		return 4;	

	// BUILDING : RGB -> 200 200 200
	if(r < 205 && g < 205 && b < 205 && r > 195 && g > 195 && b > 195)	
		return 5;	

	// POLE : RGB -> 100 100 100
	if(r < 105 && g < 105 && b < 105 && r > 95 && g > 95 && b > 95)	
		return 6;	

	// VOID
	return 0;
}

// 対象となる連結成分の環境コンテキストクラスを出力する関数
// 各連結成分で最も存在比率が高いクラスを設定
int dataset150_cclabel(IplImage *gt, IplImage *mask){
	int w = gt->width;
	int h = gt->height;

	int label_vote[7] = {0};
	int load_label;

	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			if(mask->imageData[y * mask->widthStep + x] != 0){
				load_label = dataset150_label(gt, x, y);
				label_vote[load_label]++;
			}
		}
	}

	int max_label = 0;
	int max_vote = label_vote[max_label];
	for(int i = 1 ; i < 7 ; i++){
		if(max_vote < label_vote[i]){
			max_label = i;
		}
	}

	return max_label;
}

// 環境コンテキストクラスに"Text"を追加した場合の読み取り関数
int dataset150_label_textplus(IplImage *gt, int x, int y){
	int ch = gt->nChannels;
	if(ch != 3){
		cerr << "Ground Truth Image channel Error (3ch only)" << endl;
		return -1;
	}
	int p = y * gt->widthStep + x * gt->nChannels;
	int b = static_cast<unsigned char>(gt->imageData[p]);
	int g = static_cast<unsigned char>(gt->imageData[p+1]);
	int r = static_cast<unsigned char>(gt->imageData[p+2]);

	// SKY : RGB -> 0 0 255
	if(r < 10 && g < 10 && b > 245)
		return 1;	

	// TREE&GRASS : RGB -> 0 255 0
	if(r < 10 && g > 245 && b < 10)	
		return 2;	

	// SIGN : RGB -> 255 255 0
	if(r > 245 && g > 245 && b < 10)	
		return 3;	

	// GROUND : RGB -> 255 0 0
	if(r > 245 && g < 10 && b < 10)	
		return 4;	

	// GROUND(GRASS) : RGB -> 255 0 255
	if(r > 245 && g < 10 && b > 245)	
		return 4;	

	// BUILDING : RGB -> 200 200 200
	if(r < 205 && g < 205 && b < 205 && r > 195 && g > 195 && b > 195)	
		return 5;	

	// POLE : RGB -> 100 100 100
	if(r < 105 && g < 105 && b < 105 && r > 95 && g > 95 && b > 95)	
		return 6;	

	if(r < 5 && g < 5 && b < 5)
		return 7;

	// VOID
	return 0;
}

// 対象となる連結成分の環境コンテキストクラスを出力する関数
// 各連結成分で最も存在比率が高いクラスを設定
// クラスには"text"を追加
int dataset150_cclabel_textplus(IplImage *gt, IplImage *mask){
	int w = gt->width;
	int h = gt->height;

	int label_vote[8] = {0};
	int load_label;

	for(int y = 0 ; y < h ; y++){
		for(int x = 0 ; x < w ; x++){
			if(mask->imageData[y * mask->widthStep + x] != 0){
				load_label = dataset150_label_textplus(gt, x, y);
				label_vote[load_label]++;
			}
		}
	}

	int max_label = 0;
	int max_vote = label_vote[max_label];
	for(int i = 1 ; i < 8 ; i++){
		if(max_vote < label_vote[i]){
			max_label = i;
		}
	}

	return max_label;
}


