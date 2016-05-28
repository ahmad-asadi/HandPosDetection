#include <iostream>
#include <string.h>

#include <time.h>//used to measure spending time.

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

/////////////
const int MAX_IM_SIZE = 800;

/////////////

Mat convert_color_space(Mat input)
{
	cout << "converting color space"<<endl;
	cvtColor(input, input, CV_BGR2YCrCb);
	input *= float(1)/255;

	return input ;
}

Mat down_sample(Mat input_src, int max_size)
{
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

Mat extract_gabor_filters(Mat input)
{
	cout << "Extracting Gabor Responses..." << endl ;
	clock_t start_time,stop_time;

	namedWindow("input" , WINDOW_NORMAL);
	cout << "displaying input image" <<endl;
	imshow("input" , input);

	input = down_sample(input,MAX_IM_SIZE);

	cout << "downsampled input size = " << input.rows << " * " << input.cols <<endl ;
	imshow("input" , input) ;

	input = convert_color_space(input);

	cout << "starting chronometer..." << endl;
	start_time = clock() ;

	cout << "filter convolution" <<endl;
	Mat weighted_sum_image (input.rows, input.cols, 21);
	int kernel_size = 3;
	double sig = 1, th = 0, lm = 1.0, gm = 1, ps = 1;
	for (;th <= 180 ; th += 15)
	{
		cout <<"Theta:" << th << endl;
		Mat kernel = getGaborKernel(Size(kernel_size, kernel_size), sig, th, lm, gm, ps);
		Mat filtered_image ;
		filter2D(input, filtered_image,  CV_32F, kernel);
		addWeighted(weighted_sum_image, 0.5 , filtered_image , 0.5 , 0, weighted_sum_image , weighted_sum_image.type()) ;
	}

	medianBlur(weighted_sum_image, weighted_sum_image , 3);
	
	cout << "stopping chronometer..." << endl;
	stop_time = clock();
	cout << "time spent: " << (stop_time - start_time) << " us"<<endl ;

	return weighted_sum_image;
}	
