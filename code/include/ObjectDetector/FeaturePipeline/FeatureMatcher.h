#include <opencv2/opencv.hpp>


class MatcherType{

    public:

    enum class Type{
        FLANN,
        BRUTEFORCE
    };
    

    static std::vector<MatcherType::Type> getMatcherTypes() {
        return { MatcherType::Type::FLANN, MatcherType::Type::BRUTEFORCE };
    }

   
    static std::string toString(Type type) {
        switch (type) {
            case Type::FLANN: return "FLANN";
            case Type::BRUTEFORCE: return "BRUTEFORCE";
            default: throw std::invalid_argument("No matcher type");
        }
    }
       
    private:
    Type type;

};
class FeatureMatcher{
    private:
    cv::Ptr<cv::DescriptorMatcher> features_matcher;
    MatcherType::Type type;
    void init();
    public:
    FeatureMatcher(const MatcherType::Type& type) : type{type} {this->init();}
    FeatureMatcher(const MatcherType::Type& type, cv::DescriptorMatcher* matcher) : type{type}, features_matcher{cv::Ptr<cv::DescriptorMatcher>(matcher)} {}

    void matchFeatures(const cv::Mat& queryDescriptors, const cv::Mat& modelDescriptors, std::vector<cv::DMatch>& matches) const;
    
    void updateMatcher(cv::Ptr<cv::DescriptorMatcher> new_matcher) {
        this->features_matcher.release();
        this->features_matcher = new_matcher;
    }

    const MatcherType::Type& getType() const {return type;}
    void setType(const MatcherType::Type& type) {this->type = type;}

   
};