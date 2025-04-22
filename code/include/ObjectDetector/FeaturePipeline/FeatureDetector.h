#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "Features.h"

enum class DetectorType{
    SIFT,
    ORB
};

class FeatureDetector{
    private:
    DetectorType type;
    cv::Ptr<cv::Feature2D> features_detector;

    void init();

    public:
    FeatureDetector(const DetectorType& type) : type{type} {this->init();}
    void detectFeatures(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
    void detectModelsFeatures(const std::vector<std::pair<std::string, std::string>>& models, std::vector<ModelFeatures>& models_features) const;

    void updateDetector(cv::Ptr<cv::Feature2D> new_detector) {
        this->features_detector.release();
        this->features_detector = new_detector;
    }

    const DetectorType& getType() const {return type;}
    void setType(const DetectorType& type) {this->type = type;}
};

#endif