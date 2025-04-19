#include "../include/ViolaJonesTraining.h"


int main(){
    ViolaJonesTraining training;
    std::vector<std::string> inputFolders = { "../Image_generated/Mustard_bottle_Generated", "../Image_generated/Power_drill_Generated", "../Image_generated/Sugar_box_Generated" };


    std::string outputDir = "../Image_generated";
    std::string fileName = "Mustard_bottle";
    std::string fileNameNegative = "Mustard_bottle_negative";
    int i = training.PositiveSamplesFiles(inputFolders[0], outputDir, fileName);
    int j = training.NegativeSamplesFiles({inputFolders[1], inputFolders[2]}, outputDir, fileNameNegative);

    std::cout << "Mustard_bottle Positive samples: " << i << std::endl;
    std::cout << "Mustard_bottle Negative samples: " << j << std::endl;

    fileName = "Power_drill";
    fileNameNegative = "Power_drill_negative";
    i = training.PositiveSamplesFiles(inputFolders[1], outputDir, fileName);
    j = training.NegativeSamplesFiles({inputFolders[0], inputFolders[2]}, outputDir, fileNameNegative);

    std::cout << "Power_drill Positive samples: " << i << std::endl;
    std::cout << "Power_drill Negative samples: " << j << std::endl;

    fileName = "Sugar_box";
    fileNameNegative = "Sugar_box_negative";
    i = training.PositiveSamplesFiles(inputFolders[2], outputDir, fileName);
    j = training.NegativeSamplesFiles({inputFolders[0], inputFolders[1]}, outputDir, fileNameNegative);

    std::cout << "Sugar_box Positive samples: " << i << std::endl;
    std::cout << "Sugar_box Negative samples: " << j << std::endl;

    
    
    return 0;
}

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath> // For std::max, std::min, std::fabs

// Function to calculate Intersection over Union (IoU)
double calculate_iou(const cv::Rect& rect1, const cv::Rect& rect2) {
    // Calculate intersection area
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    int intersection_area = 0;
    if (x2 > x1 && y2 > y1) {
        intersection_area = (x2 - x1) * (y2 - y1);
    }

    // Calculate union area
    int area1 = rect1.width * rect1.height;
    int area2 = rect2.width * rect2.height;
    int union_area = area1 + area2 - intersection_area;

    // Handle case where union is zero
    if (union_area <= 0) {
        return 0.0;
    }

    return static_cast<double>(intersection_area) / union_area;
}

int main(int argc, char** argv) {
    // --- 1. Set up file paths and parameters ---
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_cascade_xml> <path_to_ground_truth_file>" << std::endl;
        return -1;
    }

    std::string cascade_filepath = argv[1];
    std::string ground_truth_filepath = argv[2];

    // IoU threshold for considering a detection a True Positive match
    double iou_threshold = 0.5; // Common threshold, adjust if needed

    // Parameters for detectMultiScale - tune these for better performance
    double scaleFactor = 1.1;
    int minNeighbors = 3;
    cv::Size minSize(30, 30); // Example min detection size

    // --- 2. Load the trained cascade classifier ---
    cv::CascadeClassifier trained_cascade;
    if (!trained_cascade.load(cascade_filepath)) {
        std::cerr << "Error: Could not load cascade classifier from file: " << cascade_filepath << std::endl;
        return -1;
    }
    std::cout << "Successfully loaded cascade classifier: " << cascade_filepath << std::endl;

    // --- 3. Load Ground Truth Data ---
    std::vector<std::pair<std::string, std::vector<cv::Rect>>> test_data;
    std::ifstream gt_file(ground_truth_filepath);

    if (!gt_file.is_open()) {
        std::cerr << "Error: Could not open ground truth file: " << ground_truth_filepath << std::endl;
        return -1;
    }

    std::string line;
    while (std::getline(gt_file, line)) {
        std::stringstream ss(line);
        std::string image_path;
        int num_objects;
        ss >> image_path >> num_objects;

        std::vector<cv::Rect> gt_boxes;
        for (int i = 0; i < num_objects; ++i) {
            int x, y, w, h;
            ss >> x >> y >> w >> h;
            gt_boxes.push_back(cv::Rect(x, y, w, h));
        }
        test_data.push_back({image_path, gt_boxes});
    }
    gt_file.close();
    std::cout << "Successfully loaded ground truth data for " << test_data.size() << " images." << std::endl;

    // --- 4. Run Detection and Evaluate on Each Test Image ---
    int total_true_positives = 0;
    int total_false_positives = 0;
    int total_false_negatives = 0; // Equal to total ground truth objects minus total true positives

    std::cout << "\nStarting evaluation..." << std::endl;

    for (const auto& data : test_data) {
        std::string current_image_path = data.first;
        const std::vector<cv::Rect>& ground_truth_boxes = data.second;

        cv::Mat image = cv::imread(current_image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Warning: Could not load image: " << current_image_path << ". Skipping." << std::endl;
            continue; // Skip to the next image
        }

        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray_image, gray_image); // Optional: depends on your training setup

        // Perform detection
        std::vector<cv::Rect> detected_objects;
        trained_cascade.detectMultiScale(gray_image, detected_objects,
                                         scaleFactor, minNeighbors, 0, minSize, cv::Size());

        // --- Evaluate Detections for the current image ---
        int image_true_positives = 0;
        int image_false_positives = 0;
        int image_false_negatives = ground_truth_boxes.size(); // Start assuming all GTs are FN

        // Keep track of which ground truth boxes have been matched
        std::vector<bool> gt_matched(ground_truth_boxes.size(), false);

        // Sort detections by confidence (not directly available with Viola-Jones,
        // but minNeighbors helps filter. If using a different detector, sort by score).
        // For Viola-Jones, often sorting by area or just processing as-is is common.
        // We'll process as-is and use the IoU matching logic.

        for (const auto& detected_rect : detected_objects) {
            bool is_match = false;
            int best_gt_match_index = -1;
            double max_iou = 0.0;

            // Find the best ground truth match for this detection
            for (size_t i = 0; i < ground_truth_boxes.size(); ++i) {
                if (!gt_matched[i]) { // Only consider ground truths that haven't been matched yet
                    double iou = calculate_iou(detected_rect, ground_truth_boxes[i]);
                    if (iou > max_iou) {
                        max_iou = iou;
                        best_gt_match_index = i;
                    }
                }
            }

            // If the best match is above the IoU threshold, it's a True Positive
            if (max_iou >= iou_threshold) {
                if (best_gt_match_index != -1) { // Should always be true if max_iou >= threshold
                    image_true_positives++;
                    gt_matched[best_gt_match_index] = true; // Mark the GT as matched
                    // Since it's a TP, it's no longer an FN for this specific GT
                    image_false_negatives--;
                    is_match = true;
                }
            }

            // If a detection doesn't match any ground truth, it's a False Positive
            if (!is_match) {
                image_false_positives++;
            }
        }

        // Accumulate totals
        total_true_positives += image_true_positives;
        total_false_positives += image_false_positives;
        // False Negatives for this image were calculated by counting unmatched GTs
        total_false_negatives += image_false_negatives;

        // Optional: Print results per image
        std::cout << "Processed " << current_image_path
                  << ": Detections=" << detected_objects.size()
                  << ", GT=" << ground_truth_boxes.size()
                  << ", TP=" << image_true_positives
                  << ", FP=" << image_false_positives
                  << ", FN=" << image_false_negatives << std::endl;

        // Optional: Draw detections and GT on image for visual inspection
        /*
        cv::Mat display_image = image.clone();
        for(const auto& rect : detected_objects) cv::rectangle(display_image, rect, cv::Scalar(0, 255, 0), 2); // Green for detections
        for(const auto& rect : ground_truth_boxes) cv::rectangle(display_image, rect, cv::Scalar(0, 0, 255), 2); // Red for ground truth
        cv::imshow("Evaluation: " + current_image_path, display_image);
        cv::waitKey(0); // Wait for key press
        */
    }

    // --- 5. Calculate Overall Performance Metrics ---
    double precision = 0.0;
    if (total_true_positives + total_false_positives > 0) {
        precision = static_cast<double>(total_true_positives) / (total_true_positives + total_false_positives);
    }

    double recall = 0.0;
    // Total ground truth objects is sum of (TP + FN) across all images
    int total_ground_truth_objects = total_true_positives + total_false_negatives;
    if (total_ground_truth_objects > 0) {
        recall = static_cast<double>(total_true_positives) / total_ground_truth_objects;
    }

    // --- 6. Report Results ---
    std::cout << "\n--- Evaluation Summary ---" << std::endl;
    std::cout << "Processed " << test_data.size() << " test images." << std::endl;
    std::cout << "Total Ground Truth Objects: " << total_ground_truth_objects << std::endl;
    std::cout << "Total Detections Made: " << total_true_positives + total_false_positives << std::endl;
    std::cout << "True Positives (TP): " << total_true_positives << std::endl;
    std::cout << "False Positives (FP): " << total_false_positives << std::endl;
    std::cout << "False Negatives (FN): " << total_false_negatives << std::endl;
    std::cout << std::fixed << std::setprecision(4); // Format for metrics
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall (Hit Rate): " << recall << std::endl;
    std::cout << "IoU Threshold: " << iou_threshold << std::endl;
    std::cout << "detectMultiScale Parameters: scaleFactor=" << scaleFactor
              << ", minNeighbors=" << minNeighbors
              << ", minSize=(" << minSize.width << "," << minSize.height << ")" << std::endl;
    std::cout << "-------------------------" << std::endl;

    return 0;
}