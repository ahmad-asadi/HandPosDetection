#include<string>

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
		int scale_parameter = 40 ;

		//Assuming that input is valid...
		VideoCapture src_vid ;
		src_vid.open(string(argv[1]));

		double frame_per_second = src_vid.get(CAP_PROP_FPS);
		Size frame_size(src_vid.get(CAP_PROP_FRAME_WIDTH), src_vid.get(CAP_PROP_FRAME_HEIGHT)) ;


		VideoWriter output_stream ;
		output_stream.open(argv[2], CV_FOURCC('M','J','P','G'), frame_per_second, frame_size);

		namedWindow("Input Video", WINDOW_AUTOSIZE);
		namedWindow("Output Video", 0);

		Mat src_frame, log_polar_frame;

		while(1)
		{
				src_vid >> src_frame ;
				logPolar(src_frame , log_polar_frame, Point2f(src_frame.cols/2, src_frame.rows/2), scale_parameter, WARP_FILL_OUTLIERS) ;
				logPolar(log_polar_frame , log_polar_frame, Point2f(src_frame.cols/2, src_frame.rows/2), scale_parameter, WARP_FILL_OUTLIERS) ;

				imshow("Input Video" , src_frame);
				imshow("Output Video", log_polar_frame);

				output_stream << log_polar_frame ;
				char c = (char)waitKey(33);
				if(c == 27)
						break;
		}
		src_vid.release();
		return 0;
}
