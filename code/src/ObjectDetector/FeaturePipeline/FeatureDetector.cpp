#include "../../../include/ObjectDetector/FeaturePipeline/FeatureDetector.h"
#include "../../../include/CustomErrors.h"
#include "../../../include/Utils.h"


void FeatureDetector::detectFeatures(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {
    this->features_detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
}

void FeatureDetector::detectModelsFeatures(const std::vector<std::pair<std::string, std::string>>& models, std::vector<ModelFeatures>& models_features) const {
    models_features.clear();
    
    for (size_t idx = 0; idx < models.size(); ++idx) {

        const auto& model_pair = models[idx];

        cv::Mat img = Utils::Loader::load_image(model_pair.first);
        cv::Mat mask = Utils::Loader::load_image(model_pair.second);
     
        if (img.empty()) {
            throw CustomErrors::ImageLoadError(model_pair.first, "Error in loading image for model feature detection");
        }
        if (mask.empty()) {
            throw CustomErrors::ImageLoadError(model_pair.second, "Error in loading mask for model feature detection");
        }

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        this->features_detector->detectAndCompute(img, mask, keypoints, descriptors);

        ModelFeatures model {
            idx,        
            keypoints,
            descriptors
        };

        models_features.push_back(model);
    }
}