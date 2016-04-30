#include <iostream>
#include <string.h>

#include <time.h>//used to measure spending time.

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	clock_t start_time,stop_time;

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
	Mat input = imread(file_name);


	namedWindow("input" , WINDOW_NORMAL);
	// namedWindow("gabor" , WINDOW_AUTOSIZE);
	namedWindow("weighted_sum" , WINDOW_NORMAL);

	cout << "starting chronometer..." << endl;
	start_time = clock() ;

	cout << "displaying input image" <<endl;
	imshow("input" , input);

	cout << "converting color space"<<endl;
	cvtColor(input, input, CV_BGR2YCrCb);
	input *= float(1)/255;

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
		// imshow("gabor" , filtered_image);
		addWeighted(weighted_sum_image, 0.5 , filtered_image , 0.5 , 0, weighted_sum_image , weighted_sum_image.type()) ;
		// imshow("weighted_sum" , weighted_sum_image);
		// waitKey(0);
	}

	cout << "stopping chronometer..." << endl;
	stop_time = clock();
	cout << "time spent: " << (stop_time - start_time) << " us"<<endl ;



	cvtColor(weighted_sum_image, weighted_sum_image, CV_YCrCb2BGR);
	Mat channels[3] ;
	split(weighted_sum_image, channels);
	imshow("weighted_sum" , channels[0]);
	waitKey(0);
	imshow("weighted_sum" , channels[1]);
	waitKey(0);
	imshow("weighted_sum" , channels[2]);
	waitKey(0);


	imshow("weighted_sum" , weighted_sum_image);
	waitKey(0);

	return 0;
}	