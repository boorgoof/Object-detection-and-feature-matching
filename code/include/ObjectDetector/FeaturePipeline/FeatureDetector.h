#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "Features.h"

class FeatureDetector{
    private:
    cv::Ptr<cv::Feature2D> features_detector;

    public:
    FeatureDetector(const cv::Ptr<cv::Feature2D>& detector) : features_detector(detector) {}
    void detectFeatures(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
    void detectModelsFeatures(const std::vector<std::pair<std::string, std::string>>& models, std::vector<ModelFeatures>& models_features) const;

};

#endif