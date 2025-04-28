//Gianluca Caregnato

#include "../../../include/ObjectDetector/ViolaJones/ViolaJones.h"



ViolaJones::~ViolaJones(){}
ViolaJones::ViolaJones(const Object_Type& type){
    
    this->type = type.get_type();
    auto iter = objectTypeMap.find(this->type);
    if (iter == objectTypeMap.end()) {
        throw CustomErrors::InvalidArgumentError("ViolaJones", "Invalid object type: " + std::to_string(static_cast<int>(this->type)));
    }
    std::string cascadePath = iter->second;
    cv::CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        throw CustomErrors::FileNameError("ViolaJones", "Could not load cascade classifier from: " + cascadePath);
    }
    else {
        this->cascade = cascade;
    }

    this->set_method_name("ViolaJones");
    this->set_model_filter_name("GaussianBlur");
    this->set_test_filter_name("GaussianBlur");
}

void ViolaJones::detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) {
    
    out_labels.clear();
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
    
    if (!detections.empty()) {
        cv::Rect bestDetection = detections[0];
        for (const auto &rect : detections) {
            
            if (rect.area() > bestDetection.area()) {
                bestDetection = rect;
            }
        }
        out_labels.push_back(Label(this->type, bestDetection));        
    }
    else {
        std::cout << "WARNING: No objects detected." << std::endl;
        return;
    }
    return;
}