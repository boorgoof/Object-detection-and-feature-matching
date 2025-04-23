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

    std::ofstream log_file("log file"); 

    for (auto& obj_dataset : datasets) {

        DetectorType detector_type;
        MatcherType matcher_type;
        for (auto& d_type : detector_type.getDetectorTypes()) {
            
            for (auto& m_type : matcher_type.getMatcherTypes()) {
                //model image filter pipeline (currenlty only gaussian blur)
                ImageFilter* model_imagefilter = new ImageFilter();
                model_imagefilter->add_filter("Gaussian Blur", Filters::gaussian_blur, cv::Size(5,5));
                //test image filter pipeline (currently only gaussian blur)
                ImageFilter* test_imagefilter = new ImageFilter();
                test_imagefilter->add_filter("Gaussian Blur", Filters::gaussian_blur, cv::Size(5,5));
                //create the object detector pipeline
                ObjectDetector* object_detector = new FeaturePipeline(new FeatureDetector(d_type), new FeatureMatcher(m_type), obj_dataset.second, model_imagefilter, test_imagefilter);
        
                const Object_Type& type = obj_dataset.first;
                Dataset& ds = obj_dataset.second;
        
                std::map<std::string, std::vector<Label>> predicted_items; 
                object_detector->detect_object_whole_dataset(ds, predicted_items);
        
                std::map<std::string, std::vector<Label>> real_items = obj_dataset.second.get_test_items();

                double accuracy = Utils::DetectionAccuracy::calculateDatasetAccuracy(obj_dataset.first, real_items, predicted_items);
                double meanIoU = Utils::DetectionAccuracy::calculateMeanIoU(obj_dataset.first, real_items, predicted_items);
                

                log_file << "Objects detection pipeline: " << type.to_string() << "\n";
                log_file << "FeatureDetector: " << DetectorType::toString(d_type) << ", FeatureMatcher: " << MatcherType::toString(m_type) << "\n";
                log_file << "Accuracy: " << accuracy << "\n";
                log_file << "Mean IoU: " << meanIoU << "\n\n";

                
                std::cout << "Objects detection pipeline: " << type.to_string() << "\n";
                std::cout << "FeatureDetector: " << DetectorType::toString(d_type) << ", FeatureMatcher: " << MatcherType::toString(m_type) << "\n";
                std::cout << "Accuracy: " << accuracy << "\n";
                std::cout << "Mean IoU: " << meanIoU << "\n\n";
           
            }
                
        }

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


