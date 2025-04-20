#include "../../../include/ObjectDetector/FeaturePipeline/SIFT_FLANN_strategy.h"


void FeatureStrategy::detectModelsFeatures( const Dataset& dataset, const cv::Ptr<cv::Feature2D>& detector, std::vector<ModelFeatures>& models_features) const {
    
    for (size_t idx = 0; idx < dataset.get_models().size(); ++idx) {

        const auto& model_pair = dataset.get_models()[idx];

        cv::Mat img = Utils::Loader::load_image(model_pair.first);
        cv::Mat mask = Utils::Loader::load_image(model_pair.second);
     
        if (img.empty() || mask.empty()){
            std::cout<<"errore train o mask non caricati";
        }
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detectAndCompute(img, mask, keypoints, descriptors);

        ModelFeatures model {
            idx,        
            keypoints,
            descriptors
        };

        models_features.push_back(model);
    }
}