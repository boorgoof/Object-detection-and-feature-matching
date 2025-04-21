#include "../../../include/ObjectDetector/FeaturePipeline/SIFT_FLANN_strategy.h"


/*
1) trovi le feature di tutte le immagini in models
2) per ogni immagine nel teste le confornti con quelle in models
3) si prende il modello con piu match
4) se il match viene considerato buono, viene localizzato in un bounding box

*/

/*
ModelFeatures SIFT_FLANN_strategy::detect_and_match_best_model(const cv::Mat& query_img, const Dataset& dataset,  QueryFeatures& query_features, std::vector<cv::DMatch>& out_matches) const {
    
    std::vector<ModelFeatures> models_features;
    detectModelsFeatures(dataset,this->features_detector, models_features);
    
    //for(const auto& model : models_features){
    //    cv::Mat model_img = Utils::Loader::load_image(dataset.get_models()[model.dataset_models_idx].first);
    //    cv::Mat img_out;
    //    cv::drawKeypoints( model_img, model.keypoints, img_out );
    //    cv::imshow("model keypoints", img_out);
    //    cv::waitKey(0);
    //}
    
   
    this->features_detector->detectAndCompute(query_img,  cv::noArray(),  query_features.keypoints, query_features.descriptors); 
    //
    //cv::Mat img_out;
    //cv::drawKeypoints( query_img, query_keypoints, img_out );
    //cv::imshow("query img keypoints", img_out);
    //cv::waitKey(0);
    //
    
    std::vector<cv::DMatch> out_matches_1;
    std::vector<cv::DMatch> out_matches_2;
    //1)
    int best_model_idx = matchBestModel(query_features.descriptors, models_features, out_matches_1); // ricorda di rimettere out_matches
    if (best_model_idx < 0) {
        std::cout << "Nessun modello trovato" << std::endl;
        return ModelFeatures(); // There isn't a model that matches the query image
    }
    ModelFeatures best_model = models_features[best_model_idx];
    //2)
    int best_model_idx_2 = matchBestModel_2(query_features.descriptors, models_features, out_matches_2);
    if (best_model_idx_2 < 0) {
        std::cout << "2) Nessun modello trovato" << std::endl;
        return ModelFeatures(); // There isn't a model that matches the query image
    }
    ModelFeatures best_model_2 = models_features[best_model_idx_2];


    // 1) per controllo (da eliminare)
    cv::Mat model_img = Utils::Loader::load_image(dataset.get_models()[best_model_idx].first);
    cv::Mat img_keypoints;
    cv::drawKeypoints( model_img, best_model.keypoints, img_keypoints );
    cv::imshow("1) Miglior modello features keypoints", img_keypoints);
    cv::waitKey(0);
    // 2) per controllo (da eliminare)
    cv::Mat model_img_2 = Utils::Loader::load_image(dataset.get_models()[best_model_idx_2].first);
    cv::Mat img_keypoints_2;
    cv::drawKeypoints( model_img_2, best_model_2.keypoints, img_keypoints_2 );
    cv::imshow("2) Miglior modello features keypoints", img_keypoints_2);
    cv::waitKey(0);
    //-------------
     
    //1) per controllo (da eliminare)
    cv::Mat img_matches;
    cv::drawMatches(query_img, query_features.keypoints, model_img, best_model.keypoints, out_matches_1, img_matches);
    cv::imshow("1) Matches", img_matches);
    cv::waitKey(0);
    //-------------

    // 2) per controllo (da eliminare)
    cv::Mat img_matches_2;
    cv::drawMatches(query_img,  query_features.keypoints, model_img_2, best_model_2.keypoints, out_matches_2, img_matches_2);
    cv::imshow("2) Matches", img_matches_2);
    cv::waitKey(0);
    //-------------

    out_matches = out_matches_2;
    return best_model;
}
*/


ModelFeatures SIFT_FLANN_strategy::detect_and_match_best_model(const cv::Mat& query_img, const Dataset& dataset, QueryFeatures& query_features, std::vector<cv::DMatch>& out_matches) const {
    
    std::vector<ModelFeatures> models_features;
    detectModelsFeatures(dataset,this->features_detector, models_features);

    this->features_detector->detectAndCompute(query_img,  cv::noArray(),  query_features.keypoints, query_features.descriptors); 

    int best_model_idx = matchBestModel(query_features.descriptors, models_features, out_matches); 
    if (best_model_idx < 0) {
        return ModelFeatures(); 
    }
    ModelFeatures best_model = models_features[best_model_idx];

    return best_model;
}




int SIFT_FLANN_strategy::matchBestModel(const cv::Mat& queryDescriptors,  std::vector<ModelFeatures>& models_features, std::vector<cv::DMatch>& out_best_matches) const {
    
    int best_model_idx = -1;
    int best_score = 0;
     
    for (size_t i = 0; i < models_features.size(); ++i) {

        const auto& model = models_features[i];

        if (model.descriptors.empty()) {
            throw std::invalid_argument("Model features have not been previously completly detected");
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


int SIFT_FLANN_strategy::matchBestModel_2(const cv::Mat& queryDescriptors,  std::vector<ModelFeatures>& models_features, std::vector<cv::DMatch>& out_best_matches) const {
    
    
    int best_model_idx = -1;
    int best_score = 0;
    float distance_threshold = 150.0f; 

    for (size_t i = 0; i < models_features.size(); ++i) {

        const auto& model = models_features[i];

        if (model.descriptors.empty()) {
            throw std::invalid_argument("Model features have not been previously completly detected");
        }
    
        std::vector<std::vector<cv::DMatch>> knn_matches;
        this->features_matcher->knnMatch(queryDescriptors, model.descriptors, knn_matches, 2);  
    
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2) {
                const cv::DMatch& m = knn_matches[i][0];
                const cv::DMatch& n = knn_matches[i][1];
                if (m.distance < 0.7f * n.distance && m.distance < distance_threshold) {
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

