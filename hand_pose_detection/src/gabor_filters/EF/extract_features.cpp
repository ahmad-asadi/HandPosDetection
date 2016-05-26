#include "extract_features.h"

std::vector<KeyPoint> extract_sift_keypoints(Mat image);

void extract_features(Mat image, double* features)
{
	cout << "Extracting features..." <<endl ;
	extract_sift_keypoints(image);
}

std::vector<KeyPoint> extract_sift_keypoints(Mat image)
{
	cout << "image type: " << image.type();
	image.convertTo(image, CV_8U, 1/255 , 0) ;
	SiftFeatureDetector detector;
    vector<KeyPoint> keypoints;
    detector.detect(image, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);
    cv::imwrite("/home/asadi/sift_result.jpg", output);	
}

