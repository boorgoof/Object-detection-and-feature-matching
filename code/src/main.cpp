#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "../include/Utils.h"
#include "../include/Dataset.h"
#include "../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"



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
    std::map<Object_Type, Dataset> datasets = Utils::Loader::load_datasets(dataset_path);


    int i=0;
    for (auto& obj_dataset : datasets) {
        ObjectDetector* object_detector = new FeaturePipeline(new FeatureDetector(DetectorType::ORB), new FeatureMatcher(MatcherType::FLANN), obj_dataset.second);
        
        const Object_Type& type = obj_dataset.first;
        Dataset& ds = obj_dataset.second;

        std::map<std::string, std::vector<Label>> predicted_items; 
        object_detector->detect_object_whole_dataset(ds, predicted_items);

        std::map<std::string, std::vector<Label>> real_items = obj_dataset.second.get_test_items();
        double accuracy = Utils::DetectionAccuracy::calculateDatasetAccuracy(obj_dataset.first, real_items, predicted_items);
        std::cout << "Accuracy for " << obj_dataset.first.to_string() << ": " << accuracy << std::endl;

        double meanIoU = Utils::DetectionAccuracy::calculateMeanIoU(obj_dataset.first, real_items, predicted_items);
        std::cout << "Mean IoU for " << obj_dataset.first.to_string() << ": " << meanIoU  << std::endl;
    }



}



// distinzione mask - image
// array associativo mask - image
// moduli feature extractor, feature matcher, Dataset

//SIFT - GAUSSIAN BLUR
//ORB
//FAST
//GLOH
//FLANN - BRUTEFORCE

//BAG OF WORDS


