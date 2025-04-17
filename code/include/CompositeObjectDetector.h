#include "ObjectDetector.h"
#include <opencv2/opencv.hpp>

class CompositeObjectDetector : public ObjectDetector{
    private:
    cv::Ptr<cv::Feature2D> feature_detector;
    cv::Ptr<cv::DescriptorMatcher> feature_matcher;

    public:
    CompositeObjectDetector(cv::Ptr<cv::Feature2D> feature_detector, cv::Ptr<cv::DescriptorMatcher> feature_matcher)
        : feature_detector{feature_detector}, feature_matcher{feature_matcher} {};

    const size_t detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels);

    //add functions to extract features from image using cv::Ptr<cv::Feature2D> feature_detector

    //add functions to match features from image using cv::Ptr<cv::DescriptorMatcher> feature_matcher
};
