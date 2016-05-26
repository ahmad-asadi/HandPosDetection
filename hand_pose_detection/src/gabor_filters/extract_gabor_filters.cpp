#include <iostream>
#include <string.h>

#include <time.h>//used to measure spending time.

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

Mat extract_gabor_filters(Mat input)
{
	clock_t start_time,stop_time;

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
