#include "../ObjectDetector.h"
#include <opencv2/opencv.hpp>

class FeaturePipeline : public ObjectDetector{
    private:
    cv::Ptr<cv::Feature2D> feature_detector;
    cv::Ptr<cv::DescriptorMatcher> feature_matcher;

    public:
    FeaturePipeline(cv::Ptr<cv::Feature2D> feature_detector, cv::Ptr<cv::DescriptorMatcher> feature_matcher)
        : feature_detector{feature_detector}, feature_matcher{feature_matcher} {};

    const size_t detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) override;

    // extract features from image using cv::Ptr<cv::Feature2D> feature_detector
    void detectFeatures(const cv::Mat& img, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

    //match features from image using cv::Ptr<cv::DescriptorMatcher> feature_matcher
    void matchFeatures(const cv::Mat& queryDescriptors,const cv::Mat& trainDescriptors, std::vector<cv::DMatch>& matches) const;

    //setters
    void setFeatureDetector(const cv::Ptr<cv::Feature2D>& feature_detector);
    void setFeatureMatcher(const cv::Ptr<cv::DescriptorMatcher>& feature_matcher);
    
};
