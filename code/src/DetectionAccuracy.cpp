#include "../include/Utils.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string.h>
#include <iostream>
#include "../include/CustomErrors.h"



double Utils::DetectionAccuracy::calculateIoU(const Label& predictedLabel, const Label& realLabel) {
    
    // Insection area
    cv::Rect intersectionRect = predictedLabel.get_bounding_box() & realLabel.get_bounding_box();
    double intersection_area = intersectionRect.area();

    // Union area
    double union_area = predictedLabel.get_bounding_box().area() + realLabel.get_bounding_box().area() - intersection_area;

    // IoU
    return intersection_area / union_area;
}


double Utils::DetectionAccuracy::calculateMeanIoU(const Object_Type obj, std::map<std::string, std::vector<Label>>& realItems, const std::map<std::string, std::vector<Label>>& predictedItems){
    
    double sum = 0.0;
    int total_predictions = 0;

    for (const auto& item : realItems) {
        
        const std::string& filename = item.first;
        const std::vector<Label>& real_labels = item.second;

        // if the file isn't in the predicted map, skip it
        if(predictedItems.find(filename) == predictedItems.end()){
            continue;
        }

        const std::vector<Label>& predicted_labels = predictedItems.at(filename);

        for (const auto& real_label : real_labels) {
            
            if (real_label.get_class_name().to_string() != obj.to_string()) {
                continue;
            }

            for (const auto& predicted_label : predicted_labels) {

                if (predicted_label.get_class_name().to_string() != obj.to_string()) {
                    continue;
                }

                double iou = Utils::DetectionAccuracy::calculateIoU(predicted_label, real_label);
                sum = sum + iou;
                

            }
            total_predictions++;
        }
        
    }

    return sum / total_predictions;
}



double Utils::DetectionAccuracy::calculateDatasetAccuracy(const Object_Type obj, std::map<std::string, std::vector<Label>>& realItems, const std::map<std::string, std::vector<Label>>& predictedItems , double threshold ){
    
    double true_positive = 0.0;
    int total_items = 0;

 
    for (const auto& item : realItems) {
        
        const std::string& filename = item.first;
        const std::vector<Label>& real_labels = item.second;

        
        // if the file isn't in the predicted map, skip it
        if(predictedItems.find(filename) == predictedItems.end()){
            continue;
        }

        const std::vector<Label>& predicted_labels = predictedItems.at(filename);
        
        for (const auto& real_label : real_labels) {
            
            if (real_label.get_class_name().get_type() != obj.get_type()) {
                continue;
            }

            for (const auto& predicted_label : predicted_labels) {

                if (predicted_label.get_class_name().get_type() != obj.get_type()) {
                    continue;
                }

                double iou = Utils::DetectionAccuracy::calculateIoU(predicted_label, real_label);
                if (iou >= threshold) {
                    true_positive++;
                }
                
            }

            total_items++;
        }
        
    }

    return true_positive / total_items;
}


void Utils::DetectionAccuracy::printLabelsImg(const std::map<std::string, std::vector<Label>>& predictedItems, const std::map<std::string, std::vector<Label>>& realItems){

    
    std::string output_folder = "../Output/BoundingBox/";

    if (!std::filesystem::exists(output_folder)) {
        std::filesystem::create_directories(output_folder); 
    }

    for (const auto& item : predictedItems) {
        
        const std::string& filename = item.first;
        const std::vector<Label>& predicted_labels = item.second;

        // if the file isn't in the real map, skip it
        if(realItems.find(filename) == realItems.end()){
            continue;
        }

        const std::vector<Label>& real_labels = realItems.at(filename);
        cv::Mat img = Utils::Loader::load_image(filename);
        
        if (img.empty()) {
            throw CustomErrors::ImageLoadError("Error loading the img in printLabelsImg: " , filename);
        }

        std::string out_img_name = output_folder + Utils::Directory::get_file_basename(filename);
        cv::Mat out_img = img.clone();

        for (const auto& label : predicted_labels) {
            cv::rectangle(out_img, label.get_bounding_box(), cv::Scalar(0, 0, 255), 2);
        }

        for (const auto& label : real_labels) {
            cv::rectangle(out_img, label.get_bounding_box(), cv::Scalar(0, 255, 0), 2);
        }

        cv::imwrite(out_img_name, out_img);
    }
    
  
}