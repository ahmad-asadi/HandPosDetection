#include "extract_features.h"

/////////////////////////////////////////
Mat threshold_and_convert(Mat image);
Mat loaded_dictionary ;

/////////////////////////////////////////
const int DEBUG = 0;
const char* LEARNT_KEYPOINTS_MODEL_FILE	= "../sift_descriptors/" ;
const int KNN = 2;

/////////////////////////////////////////
struct training_pair
{
	int label;
	char file_name[128];
	char descriptor_file[256];
	Mat descriptor;
};

/////////////////////////////////////////
const char * TRAIN_ENTITY_FILE_NAME = "./.trainmodel";
std::vector<training_pair> train_entity;
std::vector<int> class_labels;
CvSVM svm ;
/////////////////////////////////////////
int extract_features(Mat image)
{

	if(train_entity.size() == 0)
	{
			// load_sift_models();
			load_dictionary();
			load_SVM();
	}

	Mat input_image = image ;
	std::vector<KeyPoint> extracted_keypoints = extract_sift_keypoints(input_image);
	Mat extracted_descriptor = compute_descriptors(input_image, extracted_keypoints);

	image *= 255;
	Mat gray_channel(image.size(), CV_8UC1) ;
	image.convertTo(gray_channel,CV_8UC1) ;

    Mat sub_mat = Mat::ones(gray_channel.size(), gray_channel.type())*255;
    subtract(sub_mat, gray_channel, gray_channel);

	std::vector<float> hog_ders = extract_hog_features(gray_channel);
	cout << "calculating feature vector..." << endl ;
	Mat feature_vector = extract_features_mat(extracted_descriptor, hog_ders);

	float predicted_label = svm.predict(feature_vector);
	cout << "PREDICTED LABEL IS: " << predicted_label << endl;
	return predicted_label;
}

void load_SVM()
{	
	cout << "Loading trained SVM, please wait..." << endl ;

	svm.load("../trained_svm") ;
}

Mat get_image_BOF(Mat image, std::vector<KeyPoint> extracted_keypoints, int KNN)
{
	cout << "Attempting to classify new image" << endl ;

	cvtColor(image, image, CV_BGR2GRAY);
	image.convertTo(image, CV_8U);

	cout << "creating description matcher" << endl ;
	Ptr<DescriptorMatcher> matcher (new FlannBasedMatcher()) ;

	cout << "creating feature detector" << endl ;
	Ptr<FeatureDetector> detector (new SiftFeatureDetector()) ;

	cout << "creating descriptor extractor" << endl ;
	Ptr<DescriptorExtractor> extractor (new SiftDescriptorExtractor()) ;

	cout << "creating bag of words image descriptor extractor" << endl ;
	BOWImgDescriptorExtractor bowDE(extractor, matcher) ;

	cout << "setting vocabulary" << endl ;
	bowDE.setVocabulary(loaded_dictionary) ;
	cout << "keypoints size:" << extracted_keypoints.size() << endl ;

	cout << "computing description matrix, image_type:" << image.type() << endl ;
	Mat bowDescriptor ;
	bowDE.compute(image, extracted_keypoints, bowDescriptor) ;

	cout << "feature mat has been created successfully." << endl ;
	imshow("bowDescriptor", bowDescriptor);
	waitKey(0);

} 

void load_dictionary()
{
	cout << "Loading trained loaded_dictionary, please wait..." << endl ;

	stringstream file_name;
	file_name << LEARNT_KEYPOINTS_MODEL_FILE << "../dictionary.yml" ;

	cout << "Reading file " << file_name.str() << endl;
	FileStorage fs(file_name.str(), FileStorage::READ) ;

	fs["vocabulary"] >> loaded_dictionary ;

	fs.release();

	cout << "dictionary has been loaded successfully." << endl ;
}


std::vector<KeyPoint> extract_sift_keypoints(Mat image)
{
	cout << "Extracting features..." << endl ;

	image = threshold_and_convert(image);

	SiftFeatureDetector detector;
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


Mat threshold_and_convert(Mat image)
{
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

Mat extract_features_mat(Mat descriptor, std::vector<float> hog_ders)
{
		int feature_size = descriptor.cols + hog_ders.size() ;
		float feature_value[feature_size] ;
		float gamma = 0.05 ;

		for(int j = 0 ; j < descriptor.cols ; j++)
		{
			feature_value[j] = 0 ;
			for(int h = 0 ; h < descriptor.rows ; h++)
			{
				float to_be_added_value = descriptor.at<float>(h,j) ;
				feature_value[j] += to_be_added_value;

				// if(feature_value[j] > 1)
				// 	feature_value[j] = 1 ;
			}	
		}


		float max = 0 ;
		for(int j = 0 ; j < descriptor.cols ; j++)
		{
			if(feature_value[j] < 0.1 )
				feature_value[j] = 0.1 ;

			feature_value[j] = 1 * pow(feature_value[j],gamma) ;

			if(max < feature_value[j])
				max = feature_value[j] ;

			if(DEBUG)
				cout << "feature_value "<< j<<": " << feature_value[j] << endl ; 
		}

		for(int j =0 ; j < descriptor.cols ; j++)
			feature_value[j] = feature_value[j] / max ; 

		// double max_hog = 0 , min_hog = 0 ;
		// for(int j = descriptor.cols ; j < descriptor.cols + hog_ders.size() ; j++)
		// {
		// 	feature_value[j] = pow(hog_ders.at(j - descriptor.cols),gamma) ;
		// 	if(feature_value[j] < 0.1 )
		// 		feature_value[j] = 0.1 ;
		// 	if(feature_value[j] < 0.1 )
		// 		feature_value[j] = 0.1 ;

		// 	if(max_hog < feature_value[j])
		// 		max_hog = feature_value[j] ;
		// 	if(min_hog > feature_value[j])
		// 		min_hog = feature_value[j] ;
		// }	

		// for(int j = descriptor.cols ; j < descriptor.cols + hog_ders.size() ; j++)
		// {
		// 	feature_value[j] = (feature_value[j]-min_hog)/(max_hog - min_hog) ;
		// 	if(DEBUG)
		// 		cout << "feature_value "<< j<<": " << feature_value[j] << endl ; 
		// }	

		max = 0 ;
		for(int j = descriptor.cols ; j < descriptor.cols + hog_ders.size() ; j++)
		{
			feature_value[j] = hog_ders.at(j - descriptor.cols) ;
			if(max < feature_value[j])
				max = feature_value[j] ;
		}	

		for(int j = descriptor.cols ; j < descriptor.cols + hog_ders.size() ; j++)
		{
			feature_value[j] = feature_value[j] / max ; 
			if(DEBUG)
				cout << "feature_value "<< j<<": " << feature_value[j] << endl ; 
		}

		Mat result (1,feature_size, CV_32FC1, feature_value) ;
		
		if(DEBUG)
			draw_image_histogram(result, 0 , 128) ;

		return result.clone() ;
}

void draw_image_histogram(Mat image, double min, double max)
{
	int hist_size = image.cols ;
	float range[] = {0, max} ;
	const float * hist_range = {range} ;
	bool uniform = true ; 
	bool accumulate = false ;

	Mat hist ;
	calcHist( &image, 1, 0, Mat(), hist, 1, &hist_size, &hist_range, uniform, accumulate );

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/hist_size );

  	Mat hist_image( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );	

  	normalize(hist, hist, 0, hist_image.rows, NORM_MINMAX, -1, Mat() );

  	for( int i = 1; i < hist_size; i++ )
  	{
    	line( hist_image, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
    	line( hist_image, Point( bin_w*(i-1), hist_h - cvRound(image.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(image.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
	}

	namedWindow("histogram of features", CV_WINDOW_AUTOSIZE );
  	imshow("histogram of features", hist_image );
  	waitKey(0);

 
}

std::vector<float> extract_hog_features(Mat image)
{
	return extract_oriented_histogram(image );

	// HOGDescriptor hog ;
	// std::vector<float> ders;
	// std::vector<Point> locs;


	// hog.compute(image,ders,Size(0,0),Size(0,4),locs);
	// cout << "ders.size: " << ders.size() << endl ;

	// return ders ;
}

std::vector<float> extract_oriented_histogram(Mat image)
{
	std::vector<float> hist;
	int total_horiz = 0 , total_vert = 0 ;
	int threshold = 100 ;
	int scale = 100 ;
	for(int i = 0 ; i < image.rows ; i ++)
	{
		float hist_horiz = 0 ;

		for(int j = 0 ; j < image.cols ; j++)
			hist_horiz += ((int)image.at<uchar>(i,j)) > threshold ? 1 : 0 ;

		hist.push_back(hist_horiz) ;
		total_horiz += hist_horiz ;
	}

	for(int j = 0 ; j < image.cols ; j++)
	{
		float hist_vert = 0 ;

		for(int i = 0 ; i < image.rows ; i++)
			hist_vert += ((int)image.at<uchar>(i,j)) > threshold ? 1 : 0 ;

		hist.push_back(hist_vert) ;
		total_vert += hist_vert ;
	}
///////////
	for(int i = 0 ; i < image.rows ; i++)
	{
		hist.at(i) = scale * (float)(hist.at(i)) / total_horiz ;
		// cout << "\t hist: " << hist.at(i) ;
	}

	for(int i = image.rows ; i < image.rows + image.cols ; i++)
	{
		hist.at(i) = scale * (float)(hist.at(i)) / total_vert ;
		// cout << "\t hist: " << hist.at(i) ;
	}

	// cout << "\n total_horiz: " << total_horiz << "\n total_vert: " << total_vert << endl ;
	// cout << "image type: " << image.type() << endl ;
	// imshow("image" , image) ;
	// waitKey(0) ;
	return hist;

}




