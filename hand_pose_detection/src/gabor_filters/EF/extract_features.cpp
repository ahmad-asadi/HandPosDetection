#include "extract_features.h"

Mat threshold_and_convert(Mat image);

void extract_features(Mat image, double* features)
{
	cout << "Extracting features..." <<endl ;
	extract_sift_keypoints(image);
	// extract_hog_features(image);
}

std::vector<KeyPoint> extract_sift_keypoints(Mat image)
{
	image = threshold_and_convert(image);

	SiftFeatureDetector detector;
    vector<KeyPoint> keypoints;
    detector.detect(image, keypoints);

    cout<<"keys:" << keypoints.size()<< endl ;

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);
    imshow("sift_keys", output);

    return keypoints;
}


void extract_hog_features(Mat image)
{
	// Size win_size(128, 256) ;
	// Size block_size(32, 32) ;
	// Size block_stride(16, 16) ;
	// Size cell_size(16, 16) ;
	// int nbins = 9;

	// vector< float> descriptors_values;
 //  	vector< Point> locations;

	// cout << "Extracting HOG features..." <<endl ;
	// Mat img_gray ;
	// cvtColor(image, img_gray, CV_RGB2GRAY);

	// namedWindow("gray" , WINDOW_NORMAL);
	// imshow("gray",img_gray);
	// waitKey(0);

	// cout <<"converting image type" << endl;
	// img_gray.convertTo(img_gray, CV_8U,1/2);

	// HOGDescriptor descriptor(win_size, block_size, block_stride, cell_size, nbins) ;

	// cout << "computing descriptors..." << endl ;
	// descriptor.compute(img_gray, descriptors_values, Size(0, 0), Size(0, 0), locations);


 //  	cout <<"descriptor number ="<< descriptors_values.size()<< ":" << endl;
  	
 //  	for (int i = 0 ; i < descriptors_values.size() ; i++)
 //  	{
 //  		if(descriptors_values.at(i) != 0)
 //  			cout << i << ") " << descriptors_values.at(i) << "\t" ;
 //  	}  
 //  	cout << "finished" << endl;
}

Mat threshold_and_convert(Mat image)
{
	threshold(image , image , 0 , 1 , 0);
	image.convertTo(image, CV_8U ,255) ;
	return image;
}

