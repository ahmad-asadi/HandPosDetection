#include <iostream>
#include <string.h>


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "EGF/extract_gabor_filters.h"
#include "EF/extract_features.h"

using namespace std;
using namespace cv;

/////////////
void show_gabor_response(Mat gabor_response);
Mat down_sample(Mat input_src, int max_size);
/////////////
int main(int argc , char** argv)
{
	cout << "Extracting Gabor Filter Set Responses..." <<endl;
 	string file_name ;
	if(argc == 1)
	{
		cout << "Enter input image file name> " ;
		cin >> file_name;
	}
	else
		file_name = argv[1] ;

	cout << "loading image..." << endl;
	Mat input_src = imread(file_name);

	Mat gabor_response = extract_gabor_filters(input_src);

	cout <<"converting gabor response's color space" <<endl;
	cvtColor(gabor_response, gabor_response, CV_YCrCb2BGR);

	show_gabor_response(gabor_response);

	cout << "splitting response image to its channels..." <<endl ;
	Mat  channels [3];
	split(gabor_response , channels);

	//TODO
	double* features;
	cout << "features 0..." << endl ;
	Mat gray_gabor_response;
	cvtColor(gabor_response, gray_gabor_response, CV_RGB2GRAY);
	extract_features(gray_gabor_response, features);
	waitKey(0);

	cout << "features 1..." << endl ;
	extract_features(channels[0], features);
	waitKey(0);

	cout << "features 2..." << endl ;
	extract_features(channels[1], features);
	waitKey(0);

	cout << "features 3..." << endl ;
	extract_features(channels[2], features);
	waitKey(0);
	return 0;
}

void show_gabor_response(Mat gabor_response)
{
	namedWindow("gabor_response" , WINDOW_NORMAL);

	Mat channels[3] ;
	split(gabor_response, channels);
	cout << "drawing channel 0 of response" <<endl;
	imshow("gabor_response" , channels[0]);
	waitKey(0);
	cout << "drawing channel 1 of response" <<endl;
	imshow("gabor_response" , channels[1]);
	waitKey(0);
	cout << "drawing channel 2 of response" <<endl;
	imshow("gabor_response" , channels[2]);
	waitKey(0);


	cout << "drawing the whole response" <<endl;
	imshow("gabor_response" , gabor_response);
	waitKey(0);
}

