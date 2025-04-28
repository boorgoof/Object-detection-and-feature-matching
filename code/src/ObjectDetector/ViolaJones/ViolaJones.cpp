#include "../../include/ObjectDetector/ViolaJones/ViolaJones.h"


ViolaJones::ViolaJones(const Object_Type::Type type){
    
    this->type = type;
    auto iter = objectTypeMap.find(this->type);
    if (iter == objectTypeMap.end()) {
        throw std::runtime_error("Cascade path not found for the given object type.");
    }
    std::string cascadePath = iter->second;
    cv::CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        throw std::runtime_error("Failed to load cascade classifier: " + cascadePath);
    }
    else {
        this->cascade = cascade;
    }

}

void ViolaJones::detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) {
    std::vector<cv::Rect> detections;
   

    
    cv::Mat gray_img;
    if (src_img.channels() == 3) {
        cv::cvtColor(src_img, gray_img, cv::COLOR_BGR2GRAY);
    } else {
        gray_img = src_img;
    }
    cv::GaussianBlur(gray_img, gray_img, cv::Size(3, 3), 0); 

    this->cascade.detectMultiScale(gray_img, detections, 1.001, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(5, 5));
    cv::Mat output_img;
    // Ensure we have a 3-channel image for drawing colored rectangles
    if (src_img.channels() == 3) {
        output_img = src_img.clone();
    } else {
        cv::cvtColor(src_img, output_img, cv::COLOR_GRAY2BGR);
    }
    if (!detections.empty()) {
        cv::Rect bestDetection = detections[0];
        for (const auto &rect : detections) {
            // Draw all detected rectangles in blue
            cv::rectangle(output_img, rect, cv::Scalar(255, 0, 0), 2);
            if (rect.area() > bestDetection.area()) {
                bestDetection = rect;
            }
        }
        // Draw the best detection in red
        cv::rectangle(output_img, bestDetection, cv::Scalar(0, 0, 255), 2);
        out_labels.emplace_back(this->type, bestDetection);
    }
    // Create the output directory if it doesn't exist
    std::filesystem::path outputDir("../OutputViolaJones");
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directory(outputDir);
    }

    std::string outputPath = (outputDir / "detected_output.jpg").string();
    cv::imwrite(outputPath, output_img);

    

    
}