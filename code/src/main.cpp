#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "../include/Utils.h"
#include "../include/Dataset.h"
#include "../include/CompositeObjectDetector.h"


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
    
    std::unique_ptr<ObjectDetector> object_detector(new CompositeObjectDetector(cv::SIFT::create(), cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED)));

    for(auto it=datasets.begin(); it != datasets.end(); ++it){
        std::vector<std::vector<Label>> output_labels;
        object_detector->detect_object_whole_dataset(it->second, output_labels);

        //analyze detection accuracy etc etc

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










