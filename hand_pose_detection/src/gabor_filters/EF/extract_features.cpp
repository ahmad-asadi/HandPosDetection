#include "extract_features.h"

/////////////////////////////////////////
Mat threshold_and_convert(Mat image);

/////////////////////////////////////////
const int DEBUG = 0;
const char* LEARNT_KEYPOINTS_MODEL_FILE	= "../sift_descriptors/" ;
/////////////////////////////////////////
std::vector<Mat> descriptors ;

/////////////////////////////////////////
void extract_features(Mat image, double* features)
{

	if(descriptors.size() == 0)
		load_sift_models();

	Mat channels [3] ;

	split(image, channels);

	std::vector<KeyPoint> extracted_keypoints = extract_sift_keypoints(channels[0]);

	Mat extracted_descriptor = compute_descriptors(channels[0], extracted_keypoints);

	std::vector<double> sift_scores = compare_sift_descriptors(extracted_descriptor) ;
	// extract_hog_features(image);
}


std::vector<double> compare_sift_descriptors(Mat img_descriptor)
{
	FlannBasedMatcher matcher;

	std::vector<double> scores;

	cout << "Comparing extracted sift detectors with thoese of trained models..." << endl ;
	for(int i = 0 ; i < descriptors.size() ; i++)
	{
		cout << "model " << i << "\t" << endl ;
		
		Mat model_descriptor = descriptors.at(i) ;
  		std::vector< DMatch > matches;

		// imshow("model_desc", model_descriptor) ;
		// moveWindow("model_desc" , 100 , 100);
		// imshow("img_desc" , img_descriptor) ;
		// moveWindow("model_desc" , 500 , 100) ;

  		model_descriptor.convertTo(model_descriptor, CV_32F);
  		cout << "model: " << model_descriptor.type() << endl;
  		cout << "img: " << img_descriptor.type() << endl;

		waitKey(0);

  		matcher.match( img_descriptor, model_descriptor, matches);

  		double avg = 0 ;
  		for(int j = 0 ; j < matches.size() ; j++)
  			avg += matches.at(i).distance;
  		avg = avg / matches.size();
  		double score = 100 * exp(-pow(avg,2)) ;
  		scores.push_back(score) ;

  		cout << "score: " << score ;
	}
}

void load_sift_models()
{

	cout << "loading learnt sift models..." << endl ;


	DIR *dir ;
	struct dirent *ent;
	if ((dir = opendir (LEARNT_KEYPOINTS_MODEL_FILE)) != NULL) {
  		while ((ent = readdir (dir)) != NULL) 
  		{
  			if(strcmp(ent->d_name ,".") == 0 || strcmp(ent->d_name ,"..")==0)
	    		continue;
	    	char full_path[100] ;
	    	strcpy(full_path , LEARNT_KEYPOINTS_MODEL_FILE);
	    	strcat(full_path, ent->d_name);
	    	cout << "reading file: " << full_path << endl;
	    	Mat descriptor = imread(full_path);
	    	cvtColor(descriptor,descriptor, CV_BGR2GRAY);
	    	descriptors.push_back(descriptor);
  		}
  		closedir (dir);
	} else {
  		cout << "Entered address does not exist: \"" << LEARNT_KEYPOINTS_MODEL_FILE << "\"" ;
  		exit(-1);
	}

	cout << "number of detected descriptors: " << descriptors.size() <<endl;

}

std::vector<KeyPoint> extract_sift_keypoints(Mat image)
{
	cout << "Extracting features..." << endl ;

	image = threshold_and_convert(image);

	SurfFeatureDetector detector(400);
    vector<KeyPoint> keypoints;
    detector.detect(image, keypoints);

    cout<<"keys:" << keypoints.size()<< endl ;

    if(DEBUG)
    {
    	// Add results to image and save.
    	cv::Mat output;
    	cv::drawKeypoints(image, keypoints, output);
    	imshow("sift_keys", output);
    	moveWindow("sift_keys" , 500 , 100);
	}
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
//	normalize(image, image, 0, 255, NORM_MINMAX, CV_8U);
	Mat new_image ;
	threshold(image , new_image , 0 , 1 , 0);
	image.convertTo(new_image, CV_8U ,255) ;
	
	if(DEBUG)
	{
		imshow("thresholded_image" , new_image );
		moveWindow("thresholded_image", 100 , 100);
	}	

	return new_image;
}

Mat compute_descriptors(Mat image, std::vector<KeyPoint> keypoints)
{
	cout << "Extracting sift descriptors..." << endl;

	SiftDescriptorExtractor extractor ;

	Mat descriptor;

	image.convertTo(image, CV_8U);

	extractor.compute(image, keypoints , descriptor);

	return descriptor;
}

