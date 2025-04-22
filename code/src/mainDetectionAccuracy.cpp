#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "../include/Utils.h"
#include "../include/Dataset.h"
#include "../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"
#include "../include/ObjectDetector/FeaturePipeline/FeatureStrategy.h"
#include "../include/ObjectDetector/FeaturePipeline/SIFT_FLANN_strategy.h"


std::map<Object_Type, Dataset> load_datasets(const std::string& dataset_path);

int main(int argc, const char* argv[]){

    std::string dataset_path = "../dataset";
    std::string output_path = "../output";

    if(argc >= 2){
        dataset_path = argv[1];
    }
    if(argc >= 3){
        output_path = argv[2];
    }
    else{
        std::cout << "NO COMMAND LINE PARAMETERS, USING DEFAULT" << std::endl;
    }

    //loads datasets' models (feature images), test images and corresponding labels
    //COLORED FEATURE IMAGES ARE NOT LOADED BUT FILE PATH IS PAIRED WITH THE CORRESPONDING MASK FILE PAHT
    //TEST IMAGES ARE NOT LOADED BUT FILE PATH IS PAIRED WITH CORRESPONDING LABEL VECTOR (that is loaded from file)
    std::map<Object_Type, Dataset> datasets = load_datasets(dataset_path);


    for (auto& obj_dataset : datasets) {

        std::map<std::string, std::vector<Label>> real_items = obj_dataset.second.get_test_items();
        std::map<std::string, std::vector<Label>> out_items = obj_dataset.second.get_test_items();

        double accuracy = Utils::DetectionAccuracy::calculateDatasetAccuracy(obj_dataset.first, real_items, out_items);
        std::cout << "Accuracy for " << obj_dataset.first.to_string() << ": " << accuracy << std::endl;

        double meanIoU = Utils::DetectionAccuracy::calculateMeanIoU(obj_dataset.first, real_items, out_items);
        std::cout << "Mean IoU for " << obj_dataset.first.to_string() << ": " << meanIoU  << std::endl;
        
    }

}

std::map<Object_Type, Dataset> load_datasets(const std::string& dataset_path){
    
    std::vector<std::string> dataset_subfolders = Utils::Directory::get_folder_filenames(dataset_path);

    std::map<Object_Type, Dataset> datasets;

    for(auto it=dataset_subfolders.begin(); it != dataset_subfolders.end(); ++it){
        
        std::vector<std::string> tokens;
        const size_t n_f = Utils::String::split_string(*it, tokens, '/');
        datasets.insert(std::pair<Object_Type, Dataset>((tokens[n_f-1]), Dataset(Object_Type(tokens[n_f-1]), *it)));
        
    }

    return datasets;
}









