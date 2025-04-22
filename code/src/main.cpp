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

    
    std::unique_ptr<FeatureStrategy> strategy = std::make_unique<SIFT_FLANN_strategy>();
    std::unique_ptr<ObjectDetector> object_detector = std::make_unique<FeaturePipeline>(std::move(strategy));
    FeaturePipeline* pipeline = dynamic_cast<FeaturePipeline*>(object_detector.get());
    
    if (!pipeline) {
        std::cerr << "Cast failed" << std::endl;
    }

    int i=0;
    for (auto& obj_dataset : datasets) {
        if (i==0){i++; continue;}

        const Object_Type& type = obj_dataset.first;
        Dataset& ds = obj_dataset.second;

        std::map<std::string, std::vector<Label>> predicted_items; 
        pipeline->detect_object_whole_dataset(ds, predicted_items);

        std::map<std::string, std::vector<Label>> real_items = obj_dataset.second.get_items_map();
        double accuracy = Utils::DetectionAccuracy::calculateDatasetAccuracy(obj_dataset.first, real_items, predicted_items);
        std::cout << "Accuracy for " << obj_dataset.first.to_string() << ": " << accuracy << std::endl;

        double meanIoU = Utils::DetectionAccuracy::calculateMeanIoU(obj_dataset.first, real_items, predicted_items);
        std::cout << "Mean IoU for " << obj_dataset.first.to_string() << ": " << meanIoU  << std::endl;

        //break;
        i++;
    }



}

std::map<Object_Type, Dataset> load_datasets(const std::string& dataset_path){
    
    std::vector<std::string> dataset_subfolders = Utils::Directory::get_folder_filenames(dataset_path);

    //REMOVING sugar box SUBFOLDER
    //dataset_subfolders.erase(dataset_subfolders.begin());

    std::map<Object_Type, Dataset> datasets;

    for(auto it=dataset_subfolders.begin(); it != dataset_subfolders.end(); ++it){
        
        std::vector<std::string> tokens;
        const size_t n_f = Utils::String::split_string(*it, tokens, '/');
        datasets.insert(std::pair<Object_Type, Dataset>((tokens[n_f-1]), Dataset(Object_Type(tokens[n_f-1]), *it)));
        
    }

    /* PRINT JUST TO CHECK IF DATASET IS LOADED CORRECTLY
    for(auto it=datasets.begin(); it != datasets.end(); ++it){

        std::cout << "DATASET" << it->second << std::endl;

        auto items = it->second.get_items();
        for(auto it2 = items.begin(); it2 != items.end(); ++it2){
            std::cout << "item: \n" <<  *it2 << std::endl;
        }

        auto models = it->second.get_models();
        for(auto it2 = models.begin(); it2 != models.end(); ++it2){
            std::cout << "model: \n" << *it2 << std::endl;
        }
    }
    */

    return datasets;
}

// distinzione mask - image
// array associativo mask - image
// moduli feature extractor, feature matcher, Dataset










