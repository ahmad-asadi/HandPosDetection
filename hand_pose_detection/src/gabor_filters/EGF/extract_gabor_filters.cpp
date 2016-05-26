
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

	// cout << "Extracting Gabor Filter Set Responses..." <<endl;
 // 	string file_name ;
	// if(argc == 1)
	// {
	// 	cout << "Enter input image file name> " ;
	// 	cin >> file_name;
	// }
	// else
	// 	file_name = argv[1] ;

	// cout << "loading image..." << endl;
	// Mat input_src = imread(file_name);


	// namedWindow("input" , WINDOW_NORMAL);
	// // namedWindow("gabor" , WINDOW_AUTOSIZE);
	// namedWindow("weighted_sum" , WINDOW_NORMAL);

	cout << "starting chronometer..." << endl;
	start_time = clock() ;

	// cout << "displaying input image" <<endl;
	// imshow("input" , input_src);

	// cout << "input downsampling ..." <<endl ;
	// cout << "input_src size = " << input_src.rows << " * " << input_src.cols <<endl ;
	// Mat input = input_src  ;
	// while(input.rows > 800 || input.cols > 800)
	// {
	// 	pyrDown(input_src,input,Size(input_src.cols/2,input_src.rows/2));
	// 	input_src = input ;
	// }

	// cout << "downsampled input size = " << input.rows << " * " << input.cols <<endl ;
	// imshow("input" , input) ;

	// cout << "converting color space"<<endl;
	// cvtColor(input, input, CV_BGR2YCrCb);
	// input *= float(1)/255;

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

	// Mat channels[3] ;
	// split(weighted_sum_image, channels);
	// cout << "drawing channel 0 of response" <<endl;
	// imshow("weighted_sum" , channels[0]);
	// waitKey(0);
	// cout << "drawing channel 1 of response" <<endl;
	// imshow("weighted_sum" , channels[1]);
	// waitKey(0);
	// cout << "drawing channel 2 of response" <<endl;
	// imshow("weighted_sum" , channels[2]);
	// waitKey(0);


	// cout << "drawing the whole response" <<endl;
	// imshow("weighted_sum" , weighted_sum_image);
	// waitKey(0);

	return weighted_sum_image;
}	