#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "../include/Utils.h"
#include "../include/Dataset.h"

int main(int argc, const char* argv[]){
    std::string path("../dataset/004_sugar_box");
    std::string label_folder("labels");
    std::string images_folder("models");

    std::string final_imgs_path = path+"/"+images_folder;
    std::string final_lbls_path = path+"/"+label_folder;

    Dataset sugarbox_dataset(Object_Type("004_sugar_box"), path);
    std::cout << sugarbox_dataset << std::endl;

    auto items = sugarbox_dataset.get_items();

    for(auto it = items.begin(); it != items.end(); ++it){
        std::cout << "item: \n" <<  (*it) << std::endl;
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










