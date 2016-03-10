#include<iostream>
#include<string>

#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"

using namespace std;
int main(int argc, char** argv)
{
		cv::namedWindow("Playing Video", cv::WINDOW_AUTOSIZE);
		cv::VideoCapture video;
		video.open(string(argv[1])) ;
		cv::Mat frame ;
		int frame_speed = 33 ;
		if(argc > 2)
				frame_speed = atoi(argv[2]);
		cout << "frame spped: " << frame_speed << ", argc:" << argc <<endl;
		while(1)
		{
				video >> frame ;
				if(!frame.data)
						break;
				cv::imshow("Playing Video" , frame);

				if(cv::waitKey(frame_speed) > 0)
						break;
		}
		return 0;
}
