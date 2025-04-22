#include <opencv2/opencv.hpp>

class FeatureMatcher{
    private:
    cv::Ptr<cv::DescriptorMatcher> features_matcher;

    public:
    FeatureMatcher(const cv::Ptr<cv::DescriptorMatcher>& matcher) : features_matcher(matcher) {}
    void matchFeatures(const cv::Mat& queryDescriptors, const cv::Mat& modelDescriptors, std::vector<cv::DMatch>& matches) const;

};