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
const int MAX_IM_SIZE = 800;

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

	Mat input = down_sample(input_src,MAX_IM_SIZE);

	cout << "downsampled input size = " << input.rows << " * " << input.cols <<endl ;
	namedWindow("input" , WINDOW_NORMAL);
	imshow("input" , input) ;

	cout << "converting color space"<<endl;
	cvtColor(input, input, CV_BGR2YCrCb);
	input *= float(1)/255;

	Mat gabor_response = extract_gabor_filters(input);

	cout <<"converting gabor response's color space" <<endl;
	cvtColor(gabor_response, gabor_response, CV_YCrCb2BGR);

	show_gabor_response(gabor_response);

	cout << "splitting response image to its channels..." <<endl ;
	Mat  channels [3];
	split(gabor_response , channels);

	//TODO
	double* features;
	cout << "features 1..." << endl ;
	extract_features(gabor_response, features);
	waitKey(0);

	// cout << "features 1..." << endl ;
	// extract_features(channels[0], features);
	// waitKey(0);

	// cout << "features 1..." << endl ;
	// extract_features(channels[0], features);
	// waitKey(0);
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


Mat down_sample(Mat input_src, int max_size)
{
	cout << "displaying input image" <<endl;
	imshow("input" , input_src);

	cout << "input downsampling ..." <<endl ;
	cout << "input_src size = " << input_src.rows << " * " << input_src.cols <<endl ;
	Mat input = input_src  ;
	while(input.rows > max_size || input.cols > max_size)
	{
		pyrDown(input_src,input,Size(input_src.cols/2,input_src.rows/2));
		input_src = input ;
	}

	return input;
}