#include <iostream>
#include <string.h>

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
void write_keypoints(std::vector<KeyPoint> keypoints);

///////////////////////
struct training_pair
{
	int label;
	char file_name[120];
};
///////////////////////
const char * KEYPOINTS_OUT_FILE_ADDR = "./.keypoints";
const int DEBUG = 0;

///////////////////////
std::vector<training_pair> train_entity;
void compute_sift_points();
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
	compute_sift_points();
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
	    			train_entity.push_back(train_ent) ;
	    			cout << "\t" << image_counter << ") Image: " << train_ent.file_name << ", Class_Label: " << train_ent.label << ", appended entities:" << train_entity.size()<<endl ;
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

	for(int i = 0 ; i < train_entity.size() ; i++)
	{
		cout << "loading image " << train_entity.at(i).file_name <<endl;
		Mat image = imread(train_entity.at(i).file_name);
		image = extract_gabor_filters(image) ;
		std::vector<KeyPoint> keypoints = extract_sift_keypoints(image);
		if(DEBUG)
			print_keypoints(keypoints);

		write_keypoints(keypoints);
	}
}

void write_keypoints(std::vector<KeyPoint> keypoints)
{
	ofstream outfile ;
	outfile.open(KEYPOINTS_OUT_FILE_ADDR, ios::out|ios::binary|ios::app);
	outfile.write((char *) (&keypoints), sizeof(keypoints));
	outfile.close();
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