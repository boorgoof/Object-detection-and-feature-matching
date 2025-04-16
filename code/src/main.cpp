#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "../include/Utils.h"
#include "../include/Dataset.h"

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

    std::vector<std::string> dataset_subfolders = Utils::Directory::get_folder_filenames(dataset_path);

    std::vector<Dataset> datasets;

    for(auto it=dataset_subfolders.begin(); it != dataset_subfolders.end(); ++it){
        std::vector<std::string> tokens;
        const size_t n_f = Utils::String::split_string(*it, tokens, '/');
        datasets.push_back(Dataset(Object_Type(tokens[n_f-1]), *it));
    }

    for(auto it=datasets.begin(); it != datasets.end(); ++it){

        std::cout << "DATASET" << *it << std::endl;

        auto items = (*it).get_items();
        for(auto it2 = items.begin(); it2 != items.end(); ++it2){
            std::cout << "item: \n" <<  (*it2) << std::endl;
        }
    }
    

    /*
    std::vector<cv::Mat> images, masks;

    size_t len = Utils::Loader::load_folder_images(final_imgs_path, images, masks);
    std::vector<std::vector<Label>> labels = Utils::Loader::load_folder_labels(final_lbls_path);
    std::cout << "model images: " << images.size() << " test labels: " << labels.size() << std::endl;
    for (auto it = labels.begin(); it != labels.end(); ++it){
        for(auto it2 = (*it).begin(); it2 != (*it).end(); ++it2){
            std::cout << (*it2) << std::endl;
        }
        std::cout << std::endl;
    }
        */
}


// distinzione mask - image
// array associativo mask - image
// moduli feature extractor, feature matcher, Dataset










