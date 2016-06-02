#include <iostream>
#include <string>
#include <sstream>

#include <dirent.h>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include "../EGF/extract_gabor_filters.h"
#include "../EF/extract_features.h"


///////////////////////
using namespace std;
using namespace cv;
///////////////////////
void analyze_dataset(char* dir_name) ;
void print_keypoints(std::vector<KeyPoint> keypoints);
void write_descriptors();
void compute_sift_points();
int check_keypoints_file_existance();

///////////////////////
struct training_pair
{
	int label;
	char file_name[128];
	char descriptor_file[256];
	Mat descriptor;
};
///////////////////////
const char * KEYPOINTS_OUT_FILE_ADDR = "../../sift_descriptors/";
const char * TRAINING_ENTITIES_FILE_NAME = "./.trainmodel";

const int DEBUG = 0;

///////////////////////
std::vector<training_pair> training_entities;
//std::vector<Mat> _descriptors;

///////////////////////
int main(int argc, char ** argv)
{
	char * dir_name ;
	if(argc >= 2)
		dir_name = argv[1] ;
	else
	{
		cout << "Enter directory address of training data:" ;
		cin >> dir_name;
	}

	analyze_dataset(dir_name);

	// if((DEBUG||!check_keypoints_file_existance()))
	compute_sift_points();

	write_descriptors();
}

int check_keypoints_file_existance()
{
	cout << "checking existance of a .keypoints file..." <<endl;
	FILE* f = fopen(KEYPOINTS_OUT_FILE_ADDR, "r");
	int res = f != NULL ;
	cout << "file " << (res ? "exists" : "does not exist. Going to generate it...") <<endl;
	return res;
}

void analyze_dataset(char* dir_name)
{
	DIR *dir , *subdir;
	int class_counter = 0 ;
	struct dirent *ent, *subent;
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
			    	
			    	training_pair train_ent ;
	    			train_ent.label = class_counter ;
	    			strcpy(train_ent.file_name , full_path) ;
	    			strcat(train_ent.file_name , "/") ;
	    			strcat(train_ent.file_name , subent->d_name) ;
	    			training_entities.push_back(train_ent) ;
	    			cout << "\t" << image_counter << ") Image: " << train_ent.file_name << ", Class_Label: " << train_ent.label << ", appended entities:" << training_entities.size()<<endl ;
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

void compute_sift_points()
{
	cout << "Analyzing train images..." << endl;

	for(int i = 0 ; i < training_entities.size() ; i++)
	{
		cout << "loading image " << training_entities.at(i).file_name <<endl;
		Mat image = imread(training_entities.at(i).file_name);
		image = extract_gabor_filters(image) ;

		Mat channels [3] ;
		split(image , channels) ;
		std::vector<KeyPoint> keypoints = extract_sift_keypoints(channels[0]);
		if(DEBUG)
			print_keypoints(keypoints);

		Mat descriptor = compute_descriptors(channels[0] , keypoints);

//		_descriptors.push_back(descriptor);
		training_entities.at(i).descriptor = descriptor;

		if(DEBUG)
		{
			imshow("descriptor" , descriptor);
			moveWindow("descriptor" , 500,500);
			waitKey(0);
		}
	}
}

void write_descriptors()
{
	// ofstream output;
	// output.open(TRAINING_ENTITIES_FILE_NAME, ios::out|ios::binary|ios::app);
	// output.write((char*)&training_entities, sizeof(training_entities));
	// output.close();

	char * file_name ;
	for(int i = 0 ; i < training_entities.size() ; i++)
	{
		Mat image_descriptor = (training_entities.at(i)).descriptor ;
		stringstream file_name;
		file_name << KEYPOINTS_OUT_FILE_ADDR << training_entities.at(i).label <<"_model_" << i << ".png" ;

		cout << "writing in file: " << file_name.str() << endl;
		// strcpy(file_name, KEYPOINTS_OUT_FILE_ADDR);
		// strcat(file_name, strcat("model_" , strcat(to_string(i).c_str(),".png"))) ;
		imwrite(file_name.str(), image_descriptor);
	}
}

void print_keypoints(std::vector<KeyPoint> keypoints)
{
	cout << "*************************\n Keypoints: " << keypoints.size() << ":" << endl;
	for (int j = 0 ; j < keypoints.size() ; j++)
	{
		cout << "KeyPoint No #" << j << ":" <<endl;
		cout << "\t\t" << "point: " << keypoints.at(j).pt << ": " << endl
				<< "\t\t" << "size" << keypoints.at(j).size << ": " << endl
				<< "\t\t" << "angle" << keypoints.at(j).angle << ": " << endl
				<< "\t\t" << "response" << keypoints.at(j).response << ": " << endl
				<< "\t\t" << "octave" << keypoints.at(j).octave << ": " << endl
				<< "\t\t" << "class_id" << keypoints.at(j).class_id << ": " << endl;
	}
}