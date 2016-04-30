#include<iostream>
#include<string>
#include<fstream>

#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;



///////////////////////////////////////////// FUNCTION DEFS

int validate_inputs(int, char**) ;	//	Checks whether inputs are valid and are well-formed or not. Returns a result < 0 in case of errors and video speed as an integer value.
int analyze_video(VideoCapture*);	//	Analyzing video and its frames to print on console, returning nubmer of frames in video.
int process_user_order(char);	//	Processes entered user order and acts according user's order; also returns false if user wnats to exit.
Mat transform(Mat, int);	//	Transforms input frame according to transformation type passed to it and returns transformed Mat instance.

///////////////////////////////////////////// GLOBALS
int slider_position = 0 ;
int run = 1 , dont_set = 0 ;
VideoCapture video;

int transformation = 0; // 0->GaussianBlur, 1->MedianBlur 
int video_speed = 33;




void onTrackBarSlide(int position , void* )
{
	video.set(CAP_PROP_POS_FRAMES, position) ;
	if(!dont_set)
			run = 1 ;
	dont_set = 0 ;
}


int main(int argc , char** argv)
{
		if( (video_speed = validate_inputs(argc, argv)) < 0)
				return -1 ;

		cout << "video speed: " << video_speed <<endl;

		namedWindow("Input", WINDOW_AUTOSIZE);	
		namedWindow("Layer2" , WINDOW_AUTOSIZE);
		namedWindow("Layer3" , WINDOW_AUTOSIZE);

		video.open(string(argv[1]));

		int frames_count = analyze_video(&video) ;

		createTrackbar("Progress" , "Input" , &slider_position , frames_count, onTrackBarSlide) ;

		Mat current_frame, layer2_frame, layer3_frame ; 

		while(1){
				if(run != 0)
				{
					video >> current_frame;

					layer2_frame = transform(current_frame, transformation);
					layer3_frame = transform(layer2_frame, transformation);
					
					if(!current_frame.data)
							break;

					int current_position = video.get(CAP_PROP_POS_FRAMES);
					dont_set = 1 ;
					setTrackbarPos("Progress", "Input" , current_position) ;
					
					imshow("Input", current_frame);
					imshow("Layer2", layer2_frame);
					imshow("Layer3", layer3_frame);

					run -= 1 ;
				}
				char user_order = (char) waitKey(video_speed) ;

				if(!process_user_order(user_order))
					break ;

		}


		return 0;
}

Mat transform(Mat input_frame , int transformation_type)
{
		Mat transformed_frame;

		switch(transformation)
		{
				case 0 :
						GaussianBlur(input_frame, transformed_frame , Size(5,5) , 3,3);
						break;
				case 1 :
						medianBlur(input_frame , transformed_frame, 5);
						break;
		}
		return transformed_frame ;
}

int validate_inputs(int argc, char** argv)
{
		int video_speed = 33 ;
		if(argc < 1)
		{
				cout << "ERROR101: Invalid Input Error: There is no video file path passed!" << endl;
				return -1 ;
		}

		if(argc == 3)
		{
				char* tmp;
				video_speed = (int) strtol(argv[2], &tmp, 10);
				if (*tmp) 
				{
						    cout << "ERROR102: Invalid Input Error: Entered video_speed should be an integer number!" <<endl;
							return -1;
				}
				
		}

		return video_speed ;
}

int analyze_video(VideoCapture* video)
{

		int frames_count = (int) video->get(CAP_PROP_FRAME_COUNT);
		int frames_width = (int) video->get(CAP_PROP_FRAME_WIDTH);
		int frames_height = (int) video->get(CAP_PROP_FRAME_HEIGHT);

		cout << "Video contains " << frames_count << " frames, all having " << frames_width << " * " << frames_height << " pixels." <<endl;

		return frames_count ;
}

int process_user_order(char user_order)
{

			if(user_order == 's')
			{
					run = 1;
					cout <<	"Single step mode" <<endl ;
			}
			if(user_order == 'r')
			{
					run = -1;
					cout << "Run mode" <<endl;
			}

			return user_order != 27 ;
}
