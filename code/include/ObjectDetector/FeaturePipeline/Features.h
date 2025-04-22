#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <opencv2/opencv.hpp>

struct ModelFeatures {
    int dataset_models_idx;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    ModelFeatures(const int models_idx, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors)
        : dataset_models_idx(models_idx), keypoints(keypoints), descriptors(descriptors) {}
    
    ModelFeatures()
    : dataset_models_idx(-1), keypoints(), descriptors() {}
};

struct QueryFeatures {

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    QueryFeatures(const int query_idx, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors)
        : keypoints(keypoints), descriptors(descriptors) {}

    QueryFeatures():  keypoints(), descriptors() {}
};

#endif