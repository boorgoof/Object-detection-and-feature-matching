#include "../../../include/ObjectDetector/FeaturePipeline/FeatureMatcher.h"

void FeatureMatcher::init(){
    switch (this->type) {
        case MatcherType::Type::FLANN:
            this->features_matcher = cv::FlannBasedMatcher::create();
            break;
        case MatcherType::Type::BRUTEFORCE:
            this->features_matcher = cv::BFMatcher::create(cv::NORM_L2);
            break;
        default:
            throw std::invalid_argument("Invalid feature matcher type");
    }
}

void FeatureMatcher::matchFeatures(const cv::Mat& queryDescriptors, const cv::Mat& modelDescriptors, std::vector<cv::DMatch>& matches) const{
    matches.clear();

    if (modelDescriptors.empty()) {
        throw std::invalid_argument("Model features have not been previously completly detected");
    }
    if (queryDescriptors.empty()) {
        throw std::invalid_argument("Query features have not been previously completly detected");
    }

    std::vector<std::vector<cv::DMatch>> knn_matches;
    this->features_matcher->knnMatch(queryDescriptors, modelDescriptors, knn_matches, 2);  

    //apply Lowe's ratio test
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2) {
            const cv::DMatch& m = knn_matches[i][0];
            const cv::DMatch& n = knn_matches[i][1];
            if (m.distance < 0.7f * n.distance) {
                matches.push_back(m);
            }
        }
    }
}