#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <dirent.h>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

void extract_features(Mat image , double* features);
std::vector<KeyPoint> extract_sift_keypoints(Mat image);
void extract_hog_features(Mat image);
void load_sift_models();
Mat compute_descriptors(Mat image, std::vector<KeyPoint> keypoints);
std::vector<double> compare_sift_descriptors(Mat img_descriptor);


