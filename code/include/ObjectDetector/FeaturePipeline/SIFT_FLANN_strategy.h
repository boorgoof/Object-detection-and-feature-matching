#ifndef SIFT_FLANN_STRATEGY_H
#define SIFT_FLANN_STRATEGY_H

#include "FeatureStrategy.h"
#include <opencv2/opencv.hpp>

class SIFT_FLANN_strategy : public FeatureStrategy {

    private:
        cv::Ptr<cv::Feature2D> features_detector;
        cv::Ptr<cv::DescriptorMatcher> features_matcher;
    
    public:

        SIFT_FLANN_strategy() : features_detector(cv::SIFT::create()),
                                features_matcher(cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::KDTreeIndexParams>(5), cv::makePtr<cv::flann::SearchParams>(50) )) {}


        ModelFeatures detect_and_match_best_model(const cv::Mat& query_img, const Dataset& dataset, QueryFeatures& query_features, std::vector<cv::DMatch>& out_matches) const override;
        int matchBestModel(const cv::Mat& queryDescriptors, std::vector<ModelFeatures>& models_features, std::vector<cv::DMatch>& out_matches) const;
        int matchBestModel_2(const cv::Mat& queryDescriptors,  std::vector<ModelFeatures>& models_features, std::vector<cv::DMatch>& out_best_matches) const;
 };



#endif // SIFT_FLANN_STRATEGY_H