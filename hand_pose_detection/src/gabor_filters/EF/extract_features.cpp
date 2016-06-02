#include "extract_features.h"

/////////////////////////////////////////
Mat threshold_and_convert(Mat image);

/////////////////////////////////////////
const int DEBUG = 1;
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

/////////////////////////////////////////
void extract_features(Mat image, double* features)
{

	if(train_entity.size() == 0)
		load_sift_models();

	Mat channels [3] ;

	split(image, channels);

	std::vector<KeyPoint> extracted_keypoints = extract_sift_keypoints(channels[0]);

	Mat extracted_descriptor = compute_descriptors(channels[0], extracted_keypoints);

	if(DEBUG)
	{
		imshow("img extracted sift descriptor", extracted_descriptor);
		moveWindow("img extracted sift descriptor" , 500,100);
		waitKey(0);
	}

	compare_sift_descriptors(extracted_descriptor, KNN) ;

	// extract_hog_features(image);
}


int compare_sift_descriptors(Mat img_descriptor, int knn)
{
	FlannBasedMatcher matcher;

	std::vector<double> scores;

	double *knn_scores = new double[knn] ;
	int *knn_labels = new int[knn] ;

	for(int i = 0 ; i < knn ; i++)
	{
		knn_scores[i] = 0 ;
		knn_labels[i] = 0 ;
	}	

	cout << "Comparing extracted sift detectors with thoese of trained models..." << endl ;
	for(int i = 0 ; i < train_entity.size() ; i++)
	{
		cout << "model " << i << ">>\t"  ;
		
		Mat model_descriptor = train_entity.at(i).descriptor ;
  		std::vector< DMatch > matches;

		if(model_descriptor.type() != CV_32F)
			model_descriptor.convertTo(model_descriptor, CV_32F);
		if(img_descriptor.type() != CV_32F)
			img_descriptor.convertTo(img_descriptor, CV_32F);


  		matcher.match( img_descriptor, model_descriptor, matches);

  		double avg = 0 ;
  		for(int j = 0 ; j < matches.size() ; j++)
  			avg += pow(matches.at(i).distance,2);
  		avg = avg / matches.size();
  		double score ;
  		// score = 100 * exp(-pow(avg,2)) ;
  		score = avg ;
  		scores.push_back(score) ;

  		cout << "score: " << score << endl;

  		int j = 0 ;
  		while(j < knn && knn_scores[j] > score)
  			j ++ ;

  		if(j < knn)
  		{
  			for (int k = knn - 1; k > j ; k--)
  			{
  				knn_scores[k] = knn_scores[k-1];
  				knn_labels[k] = knn_labels[k-1];
  			}
  			knn_scores[j] = score;
  			knn_labels[j] = train_entity.at(i).label;
  		}
	}

	cout << "***********************************" << endl;
	cout << "KNN results for k = " << knn << ":" << endl;
	for(int i = 0 ; i < knn ; i ++)
		cout << (i+1) << ") score: " << knn_scores[i] << ", label:" << knn_labels[i] << endl;
}

void load_sift_models()
{

	// cout << "loading trained model map..." << endl;
	// ifstream infile ;
	// infile.open(TRAIN_ENTITY_FILE_NAME, ios::in|ios::binary);
	// infile.read((char*)&train_entity, sizeof(train_entity));
	// infile.close();

	// cout << "existing train train_entities: " << train_entity.size() << endl ;

	// for(int i = 0 ; i < train_entity.size() ; i++)
	// {
	// 	imshow("descriptor", train_entity.at(i).descriptor);
	// 	waitKey(0);
	// }

	// for(int i = 0 ; i < train_entity.size() ; i++)
	// {
	// 	Mat dsc = imread(train_entity.at(i).descriptor_file);
	// 	cvtColor(dsc, dsc, CV_BGR2GRAY);
	// 	descriptors.push_back(dsc);
	// }

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
//	    	descriptors.push_back(descriptor);
	    	training_pair train_ent;
	    	string str (full_path) ;
	    	int slash_index = str.find_last_of("/");
	    	int underline_index = (str.substr(slash_index)).find("_");
	    	train_ent.label = atoi((str.substr(slash_index+1, underline_index - 1)).c_str());
	    	train_ent.descriptor = descriptor;
	    	train_entity.push_back(train_ent);

	    	std::vector<int>::iterator it = find(class_labels.begin(), class_labels.end(), train_ent.label);
	    	if(it == class_labels.end())
	    		class_labels.push_back(train_ent.label);

	    	cout << "full_path: " << full_path << ", label: " << train_ent.label << endl ;
  		}
  		closedir (dir);
	} else {
  		cout << "Entered address does not exist: \"" << LEARNT_KEYPOINTS_MODEL_FILE << "\"" ;
  		exit(-1);
	}

	cout << "number of detected descriptors: " << train_entity.size()
		 << ", number of detected classes: " << class_labels.size() <<endl;

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

