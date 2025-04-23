#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "Features.h"
class ImageFilter;

class DetectorType{

    public:
    enum class Type{
        SIFT,
        ORB
    };

    static std::vector<DetectorType::Type> getDetectorTypes() {
        return { DetectorType::Type::SIFT, DetectorType::Type::ORB };
    }

   
    static std::string toString(Type type) {
        switch (type) {
            case Type::SIFT: return "SIFT";
            case Type::ORB: return "ORB";
            default: throw std::invalid_argument("Unknown detector type");
        }
    }
           
    private:
    Type type;

};

class FeatureDetector{

    private:
    DetectorType::Type type;
    cv::Ptr<cv::Feature2D> features_detector;

    void init();

    public:
    FeatureDetector(const DetectorType::Type& type) : type{type} {this->init();}
    void detectFeatures(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
    void detectModelsFeatures(const std::vector<std::pair<std::string, std::string>>& models, std::vector<ModelFeatures>& models_features, ImageFilter* image_filter) const;

    void updateDetector(cv::Ptr<cv::Feature2D> new_detector) {
        this->features_detector.release();
        this->features_detector = new_detector;
    }

    const DetectorType::Type& getType() const {return type;}
    void setType(const DetectorType::Type& type) {this->type = type;}

    

};

#endif