#include <iostream>
#include <string.h>


#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "EGF/extract_gabor_filters.h"
#include "EF/extract_features.h"

using namespace std;
using namespace cv;

/////////////
struct testing_pair
{
	int label;
	char file_name[128];
	char descriptor_file[256];
	Mat descriptor;
};
/////////////
std::vector<testing_pair> testing_entities ;

/////////////
void show_gabor_response(Mat gabor_response) ;
Mat down_sample(Mat input_src, int max_size) ;
void check_data_directory(const char* dir_name) ;
/////////////
int main(int argc , char** argv)
{
	cout << "Extracting Gabor Filter Set Responses..." <<endl;
 	string file_name ;
	if(argc == 1)
	{
		cout << "Enter input image file name> " ;
		cin >> file_name;
	}
	else
		file_name = argv[1] ;

	cout << "analyzing test data directory..." << endl ;
	check_data_directory(file_name.c_str());

	int correct = 0;
	for(int i = 0 ; i < testing_entities.size() ; i ++)
	{
		file_name = testing_entities.at(i).file_name ;
		cout << "loading image..." << endl;
		Mat input_src = imread(file_name);

		Mat gabor_response = extract_gabor_filters(input_src);

		// cout <<"converting gabor response's color space" <<endl;
		// cvtColor(gabor_response, gabor_response, CV_YCrCb2BGR);

		//show_gabor_response(gabor_response);

		cout << "splitting response image to its channels..." <<endl ;
		Mat  channels [3];
		split(gabor_response , channels);

		// //TODO
		// double* features;
		// cout << "features ..." << endl ;
		// Mat gray_gabor_response;
		// cvtColor(gabor_response, gray_gabor_response, CV_RGB2GRAY);
		// extract_features(gray_gabor_response, features,	gabor_response);
		int predicted_label = extract_features(channels[0]);
		cout << "label of image " << file_name << " has been predicted as class: " << predicted_label << ", true label is: " << testing_entities.at(i).label<< endl;
		
		if(predicted_label == testing_entities.at(i).label)
			correct ++ ;

		// cout << "features 1..." << endl ;
		// extract_features(channels[0], features);
		// waitKey(0);

		// cout << "features 2..." << endl ;
		// extract_features(channels[1], features);
		// waitKey(0);

		// cout << "features 3..." << endl ;
		// extract_features(channels[2], features);
		// waitKey(0);
	}
	cout << "******************************************************************************" << endl ;
	cout << "******************* correct classification count: " << correct << " ************************" << endl ;
	cout << "*******************         accuracy: " << ((float)correct)/testing_entities.size() << "        ************************" << endl ;
	cout << "******************************************************************************" << endl ;
	return 0;
}

void show_gabor_response(Mat gabor_response)
{
	namedWindow("gabor_response" , WINDOW_NORMAL);

	Mat channels[3] ;
	split(gabor_response, channels);
	cout << "drawing channel 0 of response" <<endl;
	imshow("gabor_response" , channels[0]);
	waitKey(0);
	cout << "drawing channel 1 of response" <<endl;
	imshow("gabor_response" , channels[1]);
	waitKey(0);
	cout << "drawing channel 2 of response" <<endl;
	imshow("gabor_response" , channels[2]);
	waitKey(0);


	cout << "drawing the whole response" <<endl;
	imshow("gabor_response" , gabor_response);
	waitKey(0);
}

void check_data_directory(const char* dir_name)
{
	DIR *dir , *subdir;
	struct dirent *ent, *subent;
	int class_counter = 0 ;
	if ((dir = opendir (dir_name)) != NULL) {
  		while ((ent = readdir (dir)) != NULL) 
  		{
  			if(strcmp(ent->d_name ,".") == 0 || strcmp(ent->d_name ,"..")==0)
	    		continue;
	    	char full_path[100] ;
	    	strcpy(full_path , dir_name);
	    	strcat(full_path, ent->d_name);

	    	if((subdir = opendir(full_path))!= NULL)
	    	{
				int image_counter = 0 ;
	    		class_counter ++ ;
	    		while((subent = readdir(subdir)) != NULL)
	    		{
		  			if(strcmp(subent->d_name ,".") == 0 || strcmp(subent->d_name ,"..")==0)
	    				continue;
	    			image_counter ++ ;
			    	
			    	testing_pair test_ent ;
	    			test_ent.label = class_counter ;
	    			strcpy(test_ent.file_name , full_path) ;
	    			strcat(test_ent.file_name , "/") ;
	    			strcat(test_ent.file_name , subent->d_name) ;
	    			testing_entities.push_back(test_ent) ;
	    			cout << "\t" << image_counter << ") Image: " << test_ent.file_name << ", Class_Label: " << test_ent.label << ", appended entities:" << testing_entities.size()<<endl ;
	    		}
	    		cout << "**************************" << endl;
	    	}
	    	else
	    		cout << "Can not open directory!" << full_path <<endl ;
	    	closedir(subdir);
  		}
  		closedir (dir);
	} else {
  		cout << "Entered address does not exist: \"" << dir_name << "\"" ;
  		exit(-1);
	}

}