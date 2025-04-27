#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "../include/Utils.h"
#include "../include/Dataset.h"
#include "../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"
#include "../include/ObjectDetector/FeaturePipeline/ImageFilter.h"
#include <filesystem>


int main(int argc, const char* argv[]){

    std::string dataset_path = "../dataset";
    std::string output_path = "../output";

    if(argc >= 2){
        dataset_path = argv[1];
    }
    if(argc >= 3){
        output_path = argv[2];
        if (!std::filesystem::exists(output_path)) {
            std::filesystem::create_directories(output_path); 
        }
    }
    else{
        std::cout << "NO COMMAND LINE PARAMETERS, USING DEFAULT" << std::endl;
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

        std::map<std::string, std::vector<Label>> real_items = obj_dataset.second.get_test_items();

        //prepares the output folder for the current dataset, setting also the subfolders
        std::string output_folder = output_path + "/" + type.to_string();
        std::string log_filepath = output_folder + "/" + log_filename;
        std::string image_output_folder = output_folder + "/bounding_boxes_images/";

        if (!std::filesystem::exists(image_output_folder)) {
            std::filesystem::create_directories(image_output_folder); 
        }

        //create the log file if it doesn't exist and clear its content
        std::ofstream clear_file(log_filepath, std::ios::out);
        clear_file.close();

        for (auto& d_type : DetectorType::getDetectorTypes()) {
            
            for (auto& m_type : MatcherType::getMatcherTypes()) {
                std::map<std::string, std::vector<Label>> predicted_items;
                
                ImageFilter* model_imagefilter = new ImageFilter();
                ImageFilter* test_imagefilter = new ImageFilter();

                /*
                //model image filter pipeline (currenlty only gaussian blur)
                
                model_imagefilter->add_filter("Bilateral", Filters::bilateral_filter, 5, 75, 75);
                model_imagefilter->add_filter("CLAHE Contrast Equalization", Filters::CLAHE_contrast_equalization, 3.0, 8);
                model_imagefilter->add_filter("Unsharp Mask", Filters::unsharp_mask, 1.0, 1.5);
                
                //test image filter pipeline (currently only gaussian blur)
                test_imagefilter->add_filter("Bilateral", Filters::bilateral_filter, 5, 75, 75);
                test_imagefilter->add_filter("CLAHE Contrast Equalization", Filters::CLAHE_contrast_equalization, 3.0, 8);
                test_imagefilter->add_filter("Unsharp Mask", Filters::unsharp_mask, 1.0, 1.5);
                */
               
                //create the object detector pipeline
                //ObjectDetector* object_detector = new FeaturePipeline(new FeatureDetector(d_type), new FeatureMatcher(m_type), obj_dataset.second, model_imagefilter, test_imagefilter);
                
                ObjectDetector* object_detector = new FeaturePipeline(new FeatureDetector(d_type), new FeatureMatcher(m_type), obj_dataset.second, model_imagefilter, test_imagefilter);

                object_detector->detect_object_whole_dataset(ds, predicted_items);
                
                double accuracy = Utils::DetectionAccuracy::calculateDatasetAccuracy(obj_dataset.first, real_items, predicted_items);
                double meanIoU = Utils::DetectionAccuracy::calculateMeanIoU(obj_dataset.first, real_items, predicted_items);
                
                std::string image_output_folder_sub = image_output_folder + "/" + object_detector->get_method() + "/";

                Utils::Logger::logDetection(log_filepath, type.to_string(), object_detector->get_method(), accuracy , meanIoU );       
                Utils::Logger::printLabelsImg(image_output_folder_sub, obj_dataset.first, predicted_items, real_items);

                delete object_detector;
            }
        } 
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
