#include "../../../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"


void FeaturePipeline::detect_objects(const cv::Mat& src_img, const Dataset& dataset, std::vector<Label>& out_labels) {

   
    QueryFeatures query_features;
    std::vector<cv::DMatch> matches;
    ModelFeatures model = strategy->detect_and_match_best_model(src_img, dataset, query_features, matches);
    
    if(model.dataset_models_idx < 0){
        std::cout << "detect object: Nessun modello trovato" << std::endl; // eliminare solo il cout
        return; 
    }

    cv::Mat imgModel = Utils::Loader::load_image(dataset.get_models()[model.dataset_models_idx].first);
    cv::Mat maskModel = Utils::Loader::load_image(dataset.get_models()[model.dataset_models_idx].second);
    Label labelObj = findBoundingBox(matches, query_features.keypoints, model.keypoints, imgModel,  maskModel, src_img, dataset.get_models()[model.dataset_models_idx].first);
     
}

void FeaturePipeline::detect_objects_dataset(const std::string& query_img_name, const Dataset& dataset, std::map<std::string, std::vector<Label>>& out_items) {

    cv::Mat query_img = Utils::Loader::load_image(query_img_name);

    QueryFeatures query_features;
    std::vector<cv::DMatch> matches;
    ModelFeatures model = strategy->detect_and_match_best_model(query_img, dataset, query_features, matches);
    if(model.dataset_models_idx < 0){
        std::cout << "detect object: Nessun modello trovato" << std::endl; // eliminare solo il cout
        return; 
    }

    // DA ELIMINARE SOTTO 
    std::cout << "MAtch: " << matches.size() << std::endl;

    cv::Mat imgModel = Utils::Loader::load_image(dataset.get_models()[model.dataset_models_idx].first);
    cv::Mat maskModel = Utils::Loader::load_image(dataset.get_models()[model.dataset_models_idx].second);
    Label labelObj = findBoundingBox(matches, query_features.keypoints, model.keypoints, imgModel,  maskModel, query_img, dataset.get_models()[model.dataset_models_idx].first);
    
    out_items[query_img_name].push_back(labelObj);
}


Label FeaturePipeline::findBoundingBox(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& query_keypoint,
    const std::vector<cv::KeyPoint>& model_keypoint,
    const cv::Mat& imgModel,
    const cv::Mat& maskModel,
    const cv::Mat& imgQuery,
    Object_Type object_type) const 
{
    const int minMatches = 10;

    if (matches.size() < minMatches) {
        std::cout << "Not enough matches are found - " << matches.size() << "/" << minMatches << std::endl;
        return Label(object_type, cv::Rect());
    }

    std::vector<cv::Point2f> query_pts, model_pts;
    for (const auto& match : matches) {
        query_pts.push_back(query_keypoint[match.queryIdx].pt);
        model_pts.push_back(model_keypoint[match.trainIdx].pt);
    }

    
    cv::Rect query_rect = cv::boundingRect(query_pts);

    cv::Mat img = imgQuery.clone();
    cv::rectangle(img, query_rect, cv::Scalar(0, 255, 0), 2);

    // Togliere: è per capire se i punti sono corretti
    for (const auto& match : matches) {
        cv::circle(img, query_keypoint[match.queryIdx].pt, 5, cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow("Bounding Box on Query Image", img);
    cv::waitKey(0);

    return Label(object_type, query_rect);
}


/* METODO OPENCV MA NON VA
Label FeaturePipeline::findBoundingBox(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& query_keypoint,
    const std::vector<cv::KeyPoint>& model_keypoint,
    const cv::Mat& imgModel,
    const cv::Mat& maskModel,
    const cv::Mat& imgQuery,
    Object_Type object_type) const 
{
    const int minMatches= 10;

    if (matches.size() < minMatches) {
        std::cout << "Not enough matches are found - " << matches.size() << "/" << minMatches << std::endl;
        return Label(object_type, cv::Rect());
    }

    std::vector<cv::Point2f> query_pts, model_pts;

    for (const auto& match : matches) {
        query_pts.push_back(query_keypoint[match.queryIdx].pt);
        model_pts.push_back(model_keypoint[match.trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(model_pts, query_pts, cv::RANSAC, 5.0);
    if (H.empty()){
        std::cout << "H empty" << std::endl;
        return Label(object_type, cv::Rect());
    }
        

    cv::Rect model_rect = cv::boundingRect(maskModel); // rettangolo piu piccolo di dei pixel che non sono zero

    std::vector<cv::Point2f> model_corners = {
        {static_cast<float>(model_rect.x), static_cast<float>(model_rect.y)},
        {static_cast<float>(model_rect.x + model_rect.width), static_cast<float>(model_rect.y)},
        {static_cast<float>(model_rect.x + model_rect.width), static_cast<float>(model_rect.y + model_rect.height)},
        {static_cast<float>(model_rect.x), static_cast<float>(model_rect.y + model_rect.height)}
    };

    std::vector<cv::Point2f> scene_corners;
    cv::perspectiveTransform(model_corners, scene_corners, H);
    cv::Rect boundingBox = cv::boundingRect(scene_corners);

    cv::Mat imgQueryCopy = imgQuery.clone();
    cv::rectangle(imgQueryCopy, boundingBox, cv::Scalar(0,255,0), 2); // Disegna un rettangolo
    cv::imshow("imgQuery", imgQueryCopy);
    cv::waitKey(0);

    return Label(object_type, boundingBox);
}
*/
