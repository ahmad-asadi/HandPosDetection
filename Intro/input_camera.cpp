#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main(int argc, char** argv)
{
		namedWindow("Input Video", WINDOW_AUTOSIZE);
		VideoCapture video;
		video.open(0);
		if(!video.isOpened())
				return -1;
		Mat frame ;
		video >> frame;
		while(frame.data)
		{
			imshow("Input Video" , frame);
			char user_order = waitKey(33);
			if(user_order == 27)
					break;
			video >> frame;
		}
		return 0;
}

