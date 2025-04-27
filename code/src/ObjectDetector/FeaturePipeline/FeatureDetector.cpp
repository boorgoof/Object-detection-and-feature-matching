#include "../../../include/ObjectDetector/FeaturePipeline/FeatureDetector.h"
#include "../../../include/CustomErrors.h"
#include "../../../include/Utils.h"
#include "../../../include/ObjectDetector/FeaturePipeline/ImageFilter.h"

void FeatureDetector::init(){
    switch (this->type) {
        case DetectorType::Type::SIFT:
            this->features_detector = cv::SIFT::create();
            break;
        case DetectorType::Type::ORB:
            this->features_detector = cv::ORB::create();
            break;
        default:
            throw CustomErrors::InvalidArgumentError("type", "Invalid feature detector type");
    }
}


void FeatureDetector::detectFeatures(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {
    this->features_detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
}

void FeatureDetector::detectModelsFeatures(const std::vector<std::pair<std::string, std::string>>& models, std::vector<ModelFeatures>& models_features, ImageFilter* image_filter) const {
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

        //IF YOU WANT TO USE THE FULL IMAGE, COMMENT THE NEXT 2 LINES, otherwise the image will be cropped to the bounding box of the mask
        img = img(cv::boundingRect(mask)); // crop the image to remove the white background of the mask
        mask = mask(cv::boundingRect(mask)); // crop the mask to remove the white background of the mask

        //model image filtering if the filter component is present
        if(image_filter != nullptr){
            img = image_filter->apply_filters(img);
        }

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        this->features_detector->detectAndCompute(img, mask, keypoints, descriptors);

        ModelFeatures model {
            static_cast<int>(idx),        
            keypoints,
            descriptors
        };

        models_features.push_back(model);
    }
}