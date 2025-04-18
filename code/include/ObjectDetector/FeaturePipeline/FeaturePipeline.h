#include "../ObjectDetector.h"
#include <opencv2/opencv.hpp>

struct ModelFeatures {
    Object_Type obj;
    std::string fileName;
    cv::Mat image;
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    ModelFeatures(const Object_Type& objType, const std::string& fileName, 
                  const cv::Mat& image, const cv::Mat& mask, const std::vector<cv::KeyPoint>& keypoints, 
                  const cv::Mat& descriptors)
        : obj(objType), fileName(fileName), image(image), mask(mask), keypoints(keypoints), descriptors(descriptors) {}
};

class FeaturePipeline : public ObjectDetector{
    private:
    cv::Ptr<cv::Feature2D> feature_detector;
    cv::Ptr<cv::DescriptorMatcher> feature_matcher; 
    std::vector<ModelFeatures> models_features;

    public:
    FeaturePipeline(cv::Ptr<cv::Feature2D> feature_detector, cv::Ptr<cv::DescriptorMatcher> feature_matcher)
        : feature_detector{feature_detector}, feature_matcher{feature_matcher} {};


    const size_t detect_objects(const cv::Mat& src_img, Object_Type object_type, std::vector<Label>& out_labels) override;
    std::pair<ModelFeatures, std::vector<cv::DMatch>> selectBestModel(const cv::Mat& query_descriptors) const;
    Label findBoundingBox(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& model_keypoint, const std::vector<cv::KeyPoint>& query_keypoint, const cv::Mat& imgModel,const cv::Mat& maskModel, const cv::Mat& imgQuery, Object_Type object_type) const;

    // extract features from image using cv::Ptr<cv::Feature2D> feature_detector
    void detectFeatures(const cv::Mat& img, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

    //match features from image using cv::Ptr<cv::DescriptorMatcher> feature_matcher
    void matchFeatures(const cv::Mat& queryDescriptors,const cv::Mat& trainDescriptors, std::vector<cv::DMatch>& matches) const;
    
    
    //setters
    void setModelsfeatures(const Dataset dataset);
    void setFeatureDetector(const cv::Ptr<cv::Feature2D>& feature_detector);
    void setFeatureMatcher(const cv::Ptr<cv::DescriptorMatcher>& feature_matcher);
    
};


  /*
   DA eliminare dopo:

    1)
    SI USA Feature2D.detectAndCompute
    SI OTTIENE: std::vector<cv::KeyPoint> kp1, kp2; keypoints da immagine 1 e 2 
    
    nota:
    KeyPoint
    - float angle
    - intclass_id :object class (if the keypoints need to be clustered by an object they belong to) 
    - int octave: octave (pyramid layer) from which the keypoint has been extracted M
    - Point2f pt :coordinates of the keypoints (importante per noi)
    - float response: the response by which the most strong keypoints have been selected. 
    - float size diameter of the meaningful keypoint neighborhood 

    2) 
    SI USA DescriptorMatcher.match
    SI OTTIENE  std::vector<cv::DMatch> matches;            (solitamente si prendono i migliori) 
   
    nota:
    struct DMatch {
        int queryIdx;  indice del descrittore/keypoint nella "query" (immagine di test)
        int trainIdx;  indice del descrittore/keypoint nella "train" (model)
        float distance; distanza tra i descrittori: quindi piu piccola è meglio è.
    };

    3) 
    I MATCH VENGONO TRASFOMATI IN PUNTI
    std::vector<cv::Point2f> src_pts, dst_pts;      punti corrispondenti ai match (sono le coordinate)

    src_pts.push_back(kp1[goodMatches[i].queryIdx].pt);
    dst_pts.push_back(kp2[goodMatches[i].trainIdx].pt);

    */

