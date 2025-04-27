#include "../../../include/ObjectDetector/FeaturePipeline/FeatureMatcher.h"
#include "../../../include/CustomErrors.h"



void FeatureMatcher::init(){
    switch (this->type) {
        case MatcherType::Type::FLANN:
            this->features_matcher = cv::FlannBasedMatcher::create();
            break;
        case MatcherType::Type::BRUTEFORCE:
            this->features_matcher = cv::BFMatcher::create(cv::NORM_L2);
            break;
        default:
            throw CustomErrors::InvalidArgumentError("type", "Invalid feature matcher type");
    }
}

void FeatureMatcher::matchFeatures( const cv::Mat& modelDescriptors, const cv::Mat& sceneDescriptors, std::vector<cv::DMatch>& matches) const{
    
    matches.clear();

    if (modelDescriptors.empty()) {
        throw CustomErrors::InvalidArgumentError("modelDescriptors", "Model descriptors are empty");
    }
    if (sceneDescriptors.empty()) {
        throw CustomErrors::InvalidArgumentError("sceneDescriptors", "Scene descriptors are empty");
    }

    std::vector<std::vector<cv::DMatch>> knn_matches;
    this->features_matcher->knnMatch(modelDescriptors, sceneDescriptors, knn_matches, 2);  
    
    //apply Lowe's ratio test
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2) {
            const cv::DMatch& m = knn_matches[i][0];
            const cv::DMatch& n = knn_matches[i][1];
            if (m.distance < 0.7f * n.distance ) { 
                matches.push_back(m);
            }
        }
    }

}