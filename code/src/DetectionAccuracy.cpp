#include "../include/Utils.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string.h>
#include <iostream>
#include "../include/CustomErrors.h"



double Utils::DetectionAccuracy::calculateIoU(const Label& predictedLabel, const Label& realLabel) {
    
    // Calculate the insection area
    cv::Rect intersectionRect = predictedLabel.get_bounding_box() & realLabel.get_bounding_box();
    double intersection_area = intersectionRect.area();

    // Calculate the union area
    double union_area = predictedLabel.get_bounding_box().area() + realLabel.get_bounding_box().area() - intersection_area;

    return intersection_area / union_area;
}


double Utils::DetectionAccuracy::calculateMeanIoU(const std::map<Label, Label>& labelPairs) {
    if (labelPairs.empty()) {
        throw std::invalid_argument("Input map cannot be empty.");
    }

    double sum = 0.0;
    int count = 0;

    for (const auto& pair : labelPairs) {

        const Label& predicted = pair.first;
        const Label& real = pair.second;

        if (predicted.get_class_name().to_string() != real.get_class_name().to_string()) {
            throw std::invalid_argument("Predicted and real labels must have the same class name.");
        }

        double iou = calculateIoU(predicted, real);
        sum += iou;
        count++;
    }

    return sum / count;
}



double Utils::DetectionAccuracy::calculateAccuracy(const std::map<Label, Label>& labelPairs, double threshold ){

    if (labelPairs.empty()) {
        throw std::invalid_argument("Input map cannot be empty.");
    }

    double true_positive = 0.0;
    int total_predictions = 0;

    for (const auto& pair : labelPairs) {

        const Label& predictedLabel = pair.first;
        const Label& realLabel = pair.second;

        if (predictedLabel.get_class_name().to_string() != realLabel.get_class_name().to_string()) {
            throw std::invalid_argument("Predicted and real labels must have the same class name.");
        }

        double iou = calculateIoU(predictedLabel, realLabel);
        if (iou >= threshold ) { 
            true_positive++;
        }
        total_predictions++;
    }

    
    return true_positive / total_predictions;
}
    
