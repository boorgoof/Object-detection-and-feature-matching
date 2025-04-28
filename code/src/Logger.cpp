//Federico Meneghetti


#include "../include/Utils.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string.h>
#include <iostream>
#include "../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"
#include "../include/CustomErrors.h"


void Utils::Logger::printLabelsImg(const std::string& output_folder, const Object_Type obj, const std::map<std::string, std::vector<Label>>& predictedItems, const std::map<std::string, std::vector<Label>>& realItems){

    
    if (!std::filesystem::exists(output_folder)) {
        std::filesystem::create_directories(output_folder); 
    }

    for (const auto& item : realItems) {
        
        const std::string& filename = item.first;
        const std::vector<Label>& real_labels = item.second;

        cv::Mat img_scene = Utils::Loader::load_image(filename);
        
        if (img_scene.empty()) {
            throw CustomErrors::ImageLoadError("Error loading the img in printLabelsImg: " , filename);
        }

        std::string out_img_name = output_folder + Utils::Directory::get_file_basename(filename) + ".png";
        cv::Mat out_img = img_scene.clone();
 
        for (const auto& real_label : real_labels) {
            if (real_label.get_class_name().get_type() != obj.get_type()) {
                continue;
            }
            cv::rectangle(out_img, real_label.get_bounding_box(), cv::Scalar(0, 255, 0), 2);
        }

        if(predictedItems.find(filename) != predictedItems.end()) {

            const std::vector<Label>& predicted_labels = predictedItems.at(filename);
        
            for (const auto& predicted_label : predicted_labels) {
                if (predicted_label.get_class_name().get_type() != obj.get_type()) {
                    continue;
                }
                cv::rectangle(out_img, predicted_label.get_bounding_box(), cv::Scalar(0, 0, 255), 2);
            }
        }
              
        cv::imwrite(out_img_name, out_img);
    }
}


void Utils::Logger::logDetection(
    const std::string& file_name,
    const std::string& obj_type,
    const std::string& method_name,
    double accuracy,
    double meanIoU,
    const std::string& filter_model_name,
    const std::string& filter_scene_name) {
        
    std::ofstream log_file(file_name,  std::ios::app);

    if (!log_file.is_open()) {
        std::cerr << "Error opening log file: " << file_name << std::endl;
        return;
    }

    log_file <<  obj_type << ",";
    log_file << method_name << ",";
    log_file << filter_model_name << ",";
    log_file << filter_scene_name << ",";
    log_file << accuracy << ",";
    log_file << meanIoU << std::endl;
    
    log_file.close();
}