//	Programmed by Ahmad Asadi, Computer Vision Lab at Amirkabir University of Technology, Tehran, Iran
//	2016 Apr. 03
//	1395 Farvardin 15
//	ahmad.asadi.ir@gmail.com


#include <iostream>
#include <math.h>
#include <termios.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"



using namespace std;
using namespace cv;

Mat preprocess(Mat, int, int, float);


int main(int argc , char ** argv)
{
		const int minS = 0 ;			//0.4 * 255 ;
		const int maxS = 0.3 * 255 ;	//0.5 * 255
		const int minH = 0 ; 			//5 ;
		const int maxH = 170 ; 			//27 ;
		const float gamma = 0.04 ;

		vector<vector<Point> > contours;
  		vector<Vec4i> hierarchy;
  		RNG rng(1); 


		cout << "Creating lookup_table to use in Generalized Hough Transform (GHT)." << endl ;

		char * input_file_name ;

		if(argc > 1) 
			input_file_name = argv[1] ;
		else
		{
			cout << "Enter sample object's image file name>>" ;
			cin >> input_file_name ;
		}

		cout << "reading image from " << input_file_name << endl ;
		Mat input_image = imread(input_file_name, CV_LOAD_IMAGE_COLOR) ;

		if(!input_image.data)
		{
			cout << "cannot load image" <<endl ;
			exit(0) ;
		}
		else
			cout << "A " << input_image.rows << "*" << input_image.cols <<" image has been loaded successfully. " <<endl ;

		int rows_number = input_image.rows ;
		int cols_number = input_image.cols ;

		input_image = preprocess(input_image , rows_number , cols_number , gamma);

		namedWindow("input_image" , WINDOW_AUTOSIZE) ;
		namedWindow("contours" , WINDOW_AUTOSIZE) ;

		

		//	Transform image to HSV color space
		cvtColor(input_image, input_image, CV_BGR2HSV) ;

		cout << "Image has been transformed to HSV" << endl ;
		//cout << ">>" << input_image.at<Vec3d>(239,);
		//	Threshold image to filter skin color

		int counter = 0 ;
		for(int r=0; r<rows_number; r++){	
			for(int c=0; c<cols_number; c++) 
			{
				// Default >> 5<H<17  -   0.4<S<0.5    -   // 0.15<V<1   
				Vec3b pixel = input_image.at<Vec3b>(r,c) ;

				// if( (pixel[0]>minH) && (pixel[0] < maxH) && (pixel[1]>minS) && (pixel[1]<maxS) )// && (pixel.val(2)>(0.15*255)) && (pixel.val(2)<255) )
				if( (pixel[2]>(0.7*255)) && (pixel[2]<255) )
				{
						counter ++ ;
						// pixel[0] = 170 ;
						// pixel[1] = 255 ;
						// pixel[2] = 255 ;
				}
				else
				 	for(int i=0; i<3; ++i)	
				 		pixel[i] = 0;

				input_image.at<Vec3b>(r,c) = pixel;
			}
		}
		cout << "skin pixels: " << counter <<endl;
		cvtColor(input_image, input_image, CV_HSV2BGR);
		cvtColor(input_image, input_image, CV_BGR2GRAY);

		imwrite("../../dataset/template/out_skin_gray.jpg", input_image) ;
		threshold(input_image, input_image, 185, 195, CV_THRESH_BINARY_INV);
		
		imwrite("../../dataset/template/out_skin_gray_thresholded.jpg", input_image) ;
/*		morphologyEx(input_image, input_image, 1 , Mat1b(3,3,1), Point(-1, -1), 3);
		morphologyEx(input_image, input_image, CV_MOP_OPEN, Mat1b(7,7,1), Point(-1, -1), 1);
		morphologyEx(input_image, input_image, CV_MOP_CLOSE, Mat1b(9,9,1), Point(-1, -1), 1);

		medianBlur(input_image, input_image, 15);
*/		Canny(input_image, input_image,100 , 300 , 3 ,true);

  		/// Find contours
  		findContours( input_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  		/// Draw contours
  		Mat drawing = Mat::zeros( input_image.size(), CV_8UC3 );
  		for( int i = 0; i< contours.size(); i++ )
     	{
       		drawContours( drawing, contours, i, Scalar(255,255,255), 2, 8, hierarchy, 0, Point() );
     	}

		imshow("input_image" , input_image);
		imshow("contours" , drawing);

		waitKey(0);
		return 0;
}


Mat preprocess(Mat input_image , int rows_number , int cols_number, float gamma)
{
        Mat ycrcb;

        cvtColor(input_image,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        // split(ycrcb,channels);

//        equalizeHist(channels[0], channels[0]);
        for(int i = 0 ; i < rows_number; i++)
        	for(int j = 0 ; j < cols_number ; j++)
        		ycrcb.at<Vec3b>(i,j)[2] = 0 ;
        Mat result;
        // merge(channels,ycrcb);

        cvtColor(ycrcb,input_image,CV_YCrCb2BGR);


		// //Brightness correction
		// for(int r=0; r<rows_number; r++)
		// 	for(int c=0; c<cols_number; c++) 
		// 	{
		// 		int f = 0 ;
		// 		if(input_image.at<Vec3b>(r,c)[1] > 100)
		// 			f = 1 ;
		// 		cout <<"1)"<< (int)input_image.at<Vec3b>(r,c)[0] << ", "<< (int)input_image.at<Vec3b>(r,c)[1]<< ", "<< (int)input_image.at<Vec3b>(r,c)[2]<<endl;
		// 		input_image.at<Vec3b>(r,c)[0] = 255 * pow(((float)input_image.at<Vec3b>(r,c)[0])/255,gamma);
		// 		input_image.at<Vec3b>(r,c)[1] = 255 * pow(((float)input_image.at<Vec3b>(r,c)[1])/255,gamma);
		// 		input_image.at<Vec3b>(r,c)[2] = 255 * pow(((float)input_image.at<Vec3b>(r,c)[2])/255,gamma);
		// 		cout <<"2)"<< (int)input_image.at<Vec3b>(r,c)[0] << ", "<< (int)input_image.at<Vec3b>(r,c)[1]<< ", "<< (int)input_image.at<Vec3b>(r,c)[2]<<endl;
		// 		if(f)
		// 			getchar();
		// 	}

		imwrite("../../dataset/template/t1_preprocessed.jpg", input_image) ;

		return input_image ;
}