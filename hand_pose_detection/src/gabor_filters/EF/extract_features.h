#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <dirent.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;

int extract_features(Mat image);
std::vector<KeyPoint> extract_sift_keypoints(Mat image);
Mat compute_descriptors(Mat image, std::vector<KeyPoint> keypoints);
void load_dictionary();
Mat get_image_BOF(Mat input_image, std::vector<KeyPoint> extracted_keypoints, int KNN) ;
void load_SVM() ;
Mat extract_features_mat(Mat descriptor, std::vector<float> hog_ders) ;
void draw_image_histogram(Mat image, double min , double max) ;
std::vector<float> extract_hog_features(Mat image);
std::vector<float> extract_oriented_histogram(Mat image) ;

