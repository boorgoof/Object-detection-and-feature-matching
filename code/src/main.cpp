#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "../include/Utils.h"
#include "../include/Dataset.h"
#include "../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"
#include "../include/ImageFilter.h"
#include "../include/CustomErrors.h"
#include <filesystem>


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

    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directories(output_path); 
    }

    std::cout << "DATASET PATH: " << dataset_path << std::endl;
    std::cout << "OUTPUT PATH: " << output_path << std::endl;

    //loads datasets' models (feature images), test images and corresponding labels
    //COLORED FEATURE IMAGES ARE NOT LOADED BUT FILE PATH IS PAIRED WITH THE CORRESPONDING MASK FILE PAHT
    //TEST IMAGES ARE NOT LOADED BUT FILE PATH IS PAIRED WITH CORRESPONDING LABEL VECTOR (that is loaded from file)
    std::map<Object_Type, Dataset> datasets = Utils::Loader::load_datasets(dataset_path);

    std::string log_filename = "DetectionLog.csv";

    for (auto& obj_dataset : datasets) {

        const Object_Type& type = obj_dataset.first;
        Dataset& ds = obj_dataset.second;
       
        std::cout << "Dataset: " << type.to_string() << std::endl;

        std::map<std::string, std::vector<Label>> real_items = obj_dataset.second.get_test_items();

        //prepares the output folder for the current dataset, setting also the subfolders
        std::string output_folder = output_path + "/" + type.to_string();
        std::string log_filepath = output_folder + "/" + log_filename;
        std::string image_output_folder = output_folder + "/bounding_boxes_images/";

        if (!std::filesystem::exists(image_output_folder)) {
            std::filesystem::create_directories(image_output_folder); 
        }

        //creates the log file if it doesn't exist and clears its content
        std::ofstream clear_file(log_filepath, std::ios::out);
        clear_file << "Object_Type,Method,ModelFilter,TestFilter,Accuracy,MeanIoU" << std::endl;
        clear_file.close();


        //vector of object detectors to be used
        std::vector<ObjectDetector*> object_detectors;

        
        //PREPARE SIFT-FLANN OBJECT DETECTOR WITHOUT IMAGE FILTERS
        ObjectDetector* object_detector = new FeaturePipeline(new FeatureDetector(DetectorType::Type::SIFT), new FeatureMatcher(MatcherType::Type::FLANN), obj_dataset.second);
        object_detectors.push_back(std::move(object_detector));

        ImageFilter* model_imagefilter = new ImageFilter();
        ImageFilter* test_imagefilter = new ImageFilter();
        

        //PREPARE SIFT-FLANN OBJECT DETECTOR USING IMAGE FILTERS
        //model image filter pipeline (currenlty only gaussian blur)
        model_imagefilter->add_filter("Bilateral", Filters::bilateral_filter, 5, 75, 75);
        model_imagefilter->add_filter("CLAHE", Filters::CLAHE_contrast_equalization, 3.0, 8);
        model_imagefilter->add_filter("Unsharp", Filters::unsharp_mask, 1.0, 1.5);
        //test image filter pipeline (currently only gaussian blur)
        test_imagefilter->add_filter("Bilateral", Filters::bilateral_filter, 5, 75, 75);
        test_imagefilter->add_filter("CLAHE", Filters::CLAHE_contrast_equalization, 3.0, 8);
        test_imagefilter->add_filter("Unsharp", Filters::unsharp_mask, 1.0, 1.5);
        //create the object detector pipeline 
        object_detector = new FeaturePipeline(new FeatureDetector(DetectorType::Type::SIFT), new FeatureMatcher(MatcherType::Type::FLANN), obj_dataset.second, model_imagefilter, test_imagefilter);
        object_detectors.push_back(std::move(object_detector));

        //ADD HERE VIOLA & JONES DETECTOR

        
        //iterate over all the object detectors and detect objects in the dataset, saving the accuracy, mean IoU and the predicted items (images with bounding boxes)
        for (auto& detector : object_detectors) {

            std::map<std::string, std::vector<Label>> predicted_items;
    
            std::cout << "\tdetecting objects using " << detector->get_method() << "..." << std::endl;

            detector->detect_object_whole_dataset(ds, predicted_items);
            
            double accuracy = Utils::DetectionAccuracy::calculateDatasetAccuracy(obj_dataset.first, real_items, predicted_items);
            double meanIoU = Utils::DetectionAccuracy::calculateMeanIoU(obj_dataset.first, real_items, predicted_items);
            
            std::string image_output_folder_sub = image_output_folder + "/" + detector->get_method() + "/";

            Utils::Logger::logDetection(log_filepath, type.to_string(), detector->get_method(), accuracy, meanIoU, detector->get_filter1(), detector->get_filter2());       
            Utils::Logger::printLabelsImg(image_output_folder_sub, obj_dataset.first, predicted_items, real_items);

            std::cout << "\tend of detection using " << detector->get_method() << std::endl;
        }
        
    }

}

