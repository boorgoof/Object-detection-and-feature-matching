#include <opencv2/opencv.hpp>

enum class MatcherType{
    FLANN,
    BRUTEFORCE
};

class FeatureMatcher{
    private:
    cv::Ptr<cv::DescriptorMatcher> features_matcher;
    MatcherType type;
    void init();
    public:
    FeatureMatcher(const MatcherType& type) : type{type} {this->init();}
    FeatureMatcher(const MatcherType& type, cv::DescriptorMatcher* matcher) : type{type}, features_matcher{cv::Ptr<cv::DescriptorMatcher>(matcher)} {}

    void matchFeatures(const cv::Mat& queryDescriptors, const cv::Mat& modelDescriptors, std::vector<cv::DMatch>& matches) const;
    
    void updateMatcher(cv::Ptr<cv::DescriptorMatcher> new_matcher) {
        this->features_matcher.release();
        this->features_matcher = new_matcher;
    }

    const MatcherType& getType() const {return type;}
    void setType(const MatcherType& type) {this->type = type;}
};