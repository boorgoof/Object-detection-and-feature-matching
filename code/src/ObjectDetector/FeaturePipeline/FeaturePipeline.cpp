#include "../../../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"


const size_t FeaturePipeline::detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels){
    //add functionality using class members 
    //cv::Ptr<cv::Feature2D> feature_detector
    //cv::Ptr<cv::DescriptorMatcher> feature_matcher


    return 0;
}

void FeaturePipeline::detectFeatures(const cv::Mat& img, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {
    feature_detector->detectAndCompute(img, mask, keypoints, descriptors);
}

void FeaturePipeline::matchFeatures(const cv::Mat& queryDescriptors,const cv::Mat& trainDescriptors, std::vector<cv::DMatch>& matches) const {
    feature_matcher->match(queryDescriptors, trainDescriptors, matches);
}

void FeaturePipeline::setFeatureDetector(const cv::Ptr<cv::Feature2D>& features_detector) {
    this->feature_detector = features_detector;
}

void FeaturePipeline::setFeatureMatcher(const cv::Ptr<cv::DescriptorMatcher>& features_matcher) {
    this->feature_matcher = features_matcher;
}