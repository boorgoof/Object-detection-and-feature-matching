#include "../../../include/ObjectDetector/FeaturePipeline/SIFT_FLANN_strategy.h"


/*
1) trovi le feature di tutte le immagini in models
2) per ogni immagine nel teste le confornti con quelle in models
3) si prende il modello con piu match
4) se il match viene considerato buono, viene localizzato in un bounding box

*/

ModelFeatures SIFT_FLANN_strategy::detect_and_match_best_model(const cv::Mat& query_img, const Dataset& dataset, std::vector<cv::DMatch>& out_matches) const {
    
    std::vector<ModelFeatures> models_features;
    detectModelsFeatures(dataset,this->features_detector, models_features);
    if (models_features.empty()) {
        std::cerr << "Errore: Nessun modello disponibill matchinge per i!" << std::endl;
    }

    std::vector<cv::KeyPoint> query_keypoints;
    cv::Mat query_descriptors;
    this->features_detector->detectAndCompute(query_img,  cv::noArray(),  query_keypoints, query_descriptors); 


    int best_model_idx = matchBestModel(query_descriptors, models_features, out_matches);
    if (best_model_idx < 0) {
        std::cerr << "Errore: Nessun modello decente  per il matching" << std::endl;
        return ModelFeatures();
    }
    ModelFeatures best_model = models_features[best_model_idx];
    
     
    cv::Mat model_img = Utils::Loader::load_image(dataset.get_models()[best_model_idx].first);
    cv::Mat img_matches;

    cv::drawMatches(query_img, query_keypoints, model_img, best_model.keypoints, out_matches, img_matches);
    cv::imshow("Match tra query e modello", img_matches);
    cv::waitKey(0);

    return best_model;
}



int SIFT_FLANN_strategy::matchBestModel(const cv::Mat& queryDescriptors,  std::vector<ModelFeatures>& models_features, std::vector<cv::DMatch>& out_best_matches) const {
    
    
    int best_model_idx = -1;
    int best_score = 0;
    for (size_t i = 0; i < models_features.size(); ++i) {

        const auto& model = models_features[i];

        if (model.descriptors.empty()) {
            std::cerr << "Errore: descriptors del modello " << i << " vuoti." << std::endl;
            continue;
        }
    
        std::vector<std::vector<cv::DMatch>> knn_matches;
        this->features_matcher->knnMatch(queryDescriptors, model.descriptors, knn_matches, 2);  
    
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2) {
                const cv::DMatch& m = knn_matches[i][0];
                const cv::DMatch& n = knn_matches[i][1];
                if (m.distance < 0.7f * n.distance) {
                    good_matches.push_back(m);
                }
            }
        }

        int score = static_cast<int>(good_matches.size());

        if (score > best_score) {
            best_score = score;
            best_model_idx = static_cast<int>(i);
            out_best_matches = good_matches;
        }
    }
    return best_model_idx;
}

