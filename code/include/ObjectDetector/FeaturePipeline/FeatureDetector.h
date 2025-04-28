//Federico Meneghetti

#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "Features.h"
class ImageFilter;

/**
 * @brief Class to represent the type of a detector (e.g. SIFT, ORB, SURF).
 */
class DetectorType{

    public:

    /**
     * @brief enum to represent the different types of detectors.
     */
    enum class Type{
        SIFT,
        ORB
    };

    static std::vector<DetectorType::Type> getDetectorTypes() {
        return { DetectorType::Type::SIFT, 
            /*DetectorType::Type::ORB*/};
    }

   
    static std::string toString(Type type) {
        switch (type) {
            case Type::SIFT: return "SIFT";
            case Type::ORB: return "ORB";
            default: throw std::invalid_argument("Unknown detector type");
        }
    }
           
    private:
    /**
     * @brief the type of the detector
     */
    Type type;

};

class FeatureDetector{

    private:
    /**
     * @brief the type of the detector
     */
    DetectorType::Type type;

    /**
     * @brief the OpenCV feature detector
     */
    cv::Ptr<cv::Feature2D> features_detector;

    void init();

    public:
    FeatureDetector(const DetectorType::Type& type) : type{type} {this->init();}
    ~FeatureDetector();
    
    /**
     * @brief detect features of an image
     * @param img the image to detect features from
     * @param keypoints the output vector of keypoints
     * @param descriptors the output matrix of descriptors
     */
    void detectFeatures(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
    
     /**
     * @brief detect the features of each model in the dataset
     * @param models the vector of all models to detect features from
     * @param models_features the output vector of model features
     * @param image_filter the image filter to apply to the model image
     */
    void detectModelsFeatures(const std::vector<std::pair<std::string, std::string>>& models, std::vector<ModelFeatures>& models_features, const ImageFilter* image_filter = nullptr) const;

    void updateDetector(cv::Ptr<cv::Feature2D> new_detector) {
        this->features_detector.release();
        this->features_detector = new_detector;
    }

    const DetectorType::Type& getType() const {return type;}
    void setType(const DetectorType::Type& type) {this->type = type;}

    

};

#endif