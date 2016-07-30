#include <iostream>
#include <string>
#include <sstream>

#include <dirent.h>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"

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
void construct_BOW_kmeans(int dictionary_size, int iteration_count, float error_rate, int retries);
int check_keypoints_file_existance();
void train_svm(int iteration_count, float error_rate);

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
const int dictionary_size = 200;
const int iteration_count = 100;
const float error_rate = 1e-15;
const int retries = 1;
///////////////////////
std::vector<training_pair> training_entities;
Mat dictionary;
std::vector<double> labels_mat;
Mat training_data_mat ;

CvSVM training_svm;
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

//	construct_BOW_kmeans(dictionary_size,iteration_count,error_rate,retries);

	train_svm(iteration_count, error_rate);

	cout << "Saving trained training_svm model to ../../trained_svm file..." << endl ; 
	training_svm.save("../../trained_svm");

	cout << "Clustering has been finished successfully." << endl ;

	for(int i = 0 ; i < labels_mat.size() ; i++)
		cout << "label " << i << ": " << labels_mat.at(i) << endl ;
	write_descriptors();
}

void train_svm(int iteration_count, float error_rate)
{
	cout << "Start training\nsettting parametres..." << endl ;
	CvParamGrid gamma_grid ;
	gamma_grid.min_val = 0.000001 ;
	gamma_grid.max_val = 110 ;
	gamma_grid.step = 10 ;

	int k_fold = 10 ;

	CvSVMParams params ;
	params.svm_type = CvSVM::C_SVC ;
	params.kernel_type = CvSVM::RBF ;
	params.gamma = 3 ; 
	params.C = 0.1 ;
	params.degree = 3 ;
	params.nu = 0.9 ;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, iteration_count, error_rate);

	cout << "Creating labels mat for training..." << endl ;
	int labels_list[labels_mat.size()];
	for(int i = 0 ; i < labels_mat.size() ; i++)
	{
		labels_list[i] = labels_mat.at(i);
		cout << "final label" << labels_list[i] << endl ;
	}
	Mat training_labels_mat(labels_mat.size() , 1 , CV_32S , labels_list) ;

	cout << "Calculating PCA Projection..." << endl ;
	PCA pca(training_data_mat, Mat(), CV_PCA_DATA_AS_ROW, 0.95) ;

	cout << "training_data_mat.rows: " << training_data_mat.rows << ", training_data_mat.cols: " << training_data_mat.cols << endl ;
	cout << "Projecting training vectors to eigen space..." << endl ;
	Mat pca_training_data_mat = pca.project(training_data_mat) ; 
	cout << "PCA.rows: " << pca_training_data_mat.rows << ", PCA.cols: " << pca_training_data_mat.cols << endl ;
	cout << "PCA(0,0): " << pca_training_data_mat.at<float>(0,0) << endl ;
	namedWindow("PCA",WINDOW_AUTOSIZE);
	imshow("PCA" , pca_training_data_mat) ;
	waitKey(0) ;

	cout << "Training has been started..." << endl;
	training_svm.train_auto(pca_training_data_mat, training_labels_mat, Mat(), Mat(), params, k_fold, CvSVM::get_default_grid(CvSVM::C), gamma_grid);
	// training_svm.train_auto(training_data_mat, training_labels_mat, Mat(), Mat(), params, k_fold, CvSVM::get_default_grid(CvSVM::C), gamma_grid);

	cout << "****************************************" << endl ;
	cout << "training has been finished successfully." << endl ;
	cout << "the number of support vectors: " << training_svm.get_support_vector_count() << endl ;
	cout << "****************************************" << endl ;

	imwrite("../../training_data.png" , training_data_mat) ;
	imwrite("../../training_labels.png" , training_labels_mat) ;

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

		//Add descriptor to the dictionary
		dictionary.push_back(descriptor) ;
		channels[0]*=255;
		Mat gray_channel(channels[0].size(), CV_8UC1) ;
		channels[0].convertTo(gray_channel,CV_8UC1) ;
		cout << "calculating hog features..."<< endl ;

    	Mat sub_mat = Mat::ones(gray_channel.size(), gray_channel.type())*255;
	    subtract(sub_mat, gray_channel, gray_channel);


		std::vector<float> hog_ders = extract_hog_features(gray_channel);
		cout << "calculating feature vector..." << endl ;
		training_data_mat.push_back(extract_features_mat(descriptor,hog_ders)) ;
		labels_mat.push_back(training_entities.at(i).label);

	}

}

void construct_BOW_kmeans(int dictionary_size, int iteration_count, float error_rate, int retries)
{
	cout << "Constructing BOW_Kmeans for training images\ninitializing parameters..." << endl ;

	TermCriteria term_criteria(CV_TERMCRIT_ITER, iteration_count, error_rate) ;

	int flags = KMEANS_PP_CENTERS ; 

	cout << "Constructing kmeans trainer..." << endl ;
	BOWKMeansTrainer kmeans_trainer(dictionary_size, term_criteria, retries, flags) ;

	cout << "Clustering training images..." << endl ;
	dictionary = kmeans_trainer.cluster(dictionary) ;

	stringstream file_name;
	file_name << KEYPOINTS_OUT_FILE_ADDR << "../dictionary.yml" ;

	cout << "Creating output stream..." << endl ; 
	FileStorage fs(file_name.str(), FileStorage::WRITE) ;

	cout << "Writing dictionary to " << file_name.str() << " file..." << endl ;
	fs << "vocabulary" << dictionary ;

	fs.release() ;

}


void write_descriptors()
{

	char * file_name ;
	for(int i = 0 ; i < training_entities.size() ; i++)
	{
		Mat image_descriptor = (training_entities.at(i)).descriptor ;
		stringstream file_name;
		file_name << KEYPOINTS_OUT_FILE_ADDR << training_entities.at(i).label <<"_model_" << i << ".png" ;

		cout << "writing in file: " << file_name.str() << endl;
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