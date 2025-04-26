#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "../include/Utils.h"
#include "../include/Dataset.h"
#include "../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"
#include "../include/ObjectDetector/FeaturePipeline/ImageFilter.h"



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

    std::string log_filename = "../Output/DetectionLog.txt";
    std::ofstream clear_file(log_filename, std::ios::out);
    clear_file.close();


    for (auto& obj_dataset : datasets) {

        const Object_Type& type = obj_dataset.first;
        Dataset& ds = obj_dataset.second;

        std::map<std::string, std::vector<Label>> predicted_items; 
        std::map<std::string, std::vector<Label>> real_items = obj_dataset.second.get_test_items();

        ObjectDetector* object_detector = new FeaturePipeline(new FeatureDetector(DetectorType::Type::SIFT), new FeatureMatcher(MatcherType::Type::FLANN), obj_dataset.second);
        object_detector->detect_object_whole_dataset(ds, predicted_items);
        
        double accuracy = Utils::DetectionAccuracy::calculateDatasetAccuracy(obj_dataset.first, real_items, predicted_items);
        double meanIoU = Utils::DetectionAccuracy::calculateMeanIoU(obj_dataset.first, real_items, predicted_items);
        
        Utils::Logger::logDetection( log_filename, type.to_string(), DetectorType::toString(DetectorType::Type::SIFT), MatcherType::toString(MatcherType::Type::FLANN), accuracy , meanIoU );       
        Utils::Logger::printLabelsImg(obj_dataset.first, predicted_items, real_items);    
    }
   
}

/*
for (auto& obj_dataset : datasets) {

        std::map<std::string, std::vector<Label>> predicted_items; 
        std::map<std::string, std::vector<Label>> real_items = obj_dataset.second.get_test_items();

        double best_accuracy = 0.0;
        std::map<std::string, std::vector<Label>> best_predicted_items; 

        const Object_Type& type = obj_dataset.first;
        Dataset& ds = obj_dataset.second;

        DetectorType detector_type;
        MatcherType matcher_type;

        for (auto& d_type : detector_type.getDetectorTypes()) {
            
            for (auto& m_type : matcher_type.getMatcherTypes()) {

                
                //model image filter pipeline (currenlty only gaussian blur)
                //ImageFilter* model_imagefilter = new ImageFilter();
                //model_imagefilter->add_filter("Gaussian Blur", Filters::gaussian_blur, cv::Size(5,5));
                
                //test image filter pipeline (currently only gaussian blur)
                //ImageFilter* test_imagefilter = new ImageFilter();
                //test_imagefilter->add_filter("Gaussian Blur", Filters::gaussian_blur, cv::Size(5,5));
                
                //create the object detector pipeline
                //ObjectDetector* object_detector = new FeaturePipeline(new FeatureDetector(d_type), new FeatureMatcher(m_type), obj_dataset.second, model_imagefilter, test_imagefilter);
                

                ObjectDetector* object_detector = new FeaturePipeline(new FeatureDetector(d_type), new FeatureMatcher(m_type), obj_dataset.second);

                object_detector->detect_object_whole_dataset(ds, predicted_items);
                
                
                double accuracy = Utils::DetectionAccuracy::calculateDatasetAccuracy(obj_dataset.first, real_items, predicted_items);
                double meanIoU = Utils::DetectionAccuracy::calculateMeanIoU(obj_dataset.first, real_items, predicted_items);
                
                if (accuracy > best_accuracy) {
                    best_accuracy = accuracy;
                    best_predicted_items = predicted_items;
                }

                Utils::Logger::logDetection( log_filename, type.to_string(), DetectorType::toString(d_type), MatcherType::toString(m_type), accuracy , meanIoU );
                   
            }    
        }

        Utils::Logger::printLabelsImg(obj_dataset.first, best_predicted_items, real_items);    
    }

*/
