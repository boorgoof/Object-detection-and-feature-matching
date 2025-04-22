#ifndef FEATURESTRATEGY_H
#define FEATURESTRATEGY_H

#include <opencv2/opencv.hpp>
#include "../../Label.h"
#include "../../Dataset.h"

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
/*
enum Scope{
    MODEL,
    QUERY
}

Features{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Scope scope;

}*/

class FeatureStrategy {

    public:
        virtual ~FeatureStrategy() = default;
        virtual ModelFeatures detect_and_match_best_model(const cv::Mat& query_img, const Dataset& dataset, QueryFeatures& query_features, std::vector<cv::DMatch>& out_matches) const = 0;

    protected:

        void detectModelsFeatures( const Dataset& dataset, const cv::Ptr<cv::Feature2D>& detector, std::vector<ModelFeatures>&  models_features) const;
       
};


#endif // FEATURESTRATEGY_H
