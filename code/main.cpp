#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "Dataset.h"

int main(int argc, const char* argv[]){
    std::string path("../dataset/004_sugar_box");
    std::string label_folder("labels");
    std::string images_folder("test_images");

    std::string final_imgs_path = path+"/"+images_folder;
    std::string final_lbls_path = path+"/"+label_folder;

    std::vector<cv::Mat> images = LoaderUtils::load_folder_images(final_imgs_path);
    std::vector<Label> labels = LoaderUtils::load_folder_labels(final_lbls_path);
    std::cout << images.size() << " " << labels.size() << std::endl;
}