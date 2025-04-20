#include "../../../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"


void FeaturePipeline::detect_objects(const cv::Mat& src_img, const Dataset& dataset, std::vector<Label>& out_labels) {

    std::vector<cv::DMatch> matches;
    
    ModelFeatures model = strategy->detect_and_match_best_model(src_img, dataset, matches);
    
    cv::Mat maskModel = Utils::Loader::load_image(dataset.get_models()[model.dataset_models_idx].second);
    //findBoundingBox_1(matches, model.keypoints, model.keypoints, src_img,  maskModel, src_img, dataset.get_models()[model.dataset_models_idx].first);
     
}


Label FeaturePipeline::findBoundingBox_1(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& query_keypoint,
    const std::vector<cv::KeyPoint>& model_keypoint,
    const cv::Mat& imgModel,
    const cv::Mat& maskModel,
    const cv::Mat& imgQuery,
    Object_Type object_type) const 
{
    const int minMatches= 10;

    if (matches.size() < minMatches) {
        return Label(object_type, cv::Rect());
    }

    std::vector<cv::Point2f> query_pts, model_pts;

    for (const auto& match : matches) {
        query_pts.push_back(query_keypoint[match.queryIdx].pt);
        model_pts.push_back(model_keypoint[match.trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(model_pts, query_pts, cv::RANSAC, 5.0);
    if (H.empty())
        std::cout << "H empty" << std::endl;
        return Label(object_type, cv::Rect());

    cv::Rect model_rect = cv::boundingRect(maskModel); // rettangolo piu piccolo di dei pixel che non sono zero

    std::vector<cv::Point2f> model_corners = {
        {(model_rect.x), (model_rect.y)},
        {(model_rect.x + model_rect.width), (model_rect.y)},
        {(model_rect.x + model_rect.width), (model_rect.y + model_rect.height)},
        {(model_rect.x), (model_rect.y + model_rect.height)}
    };

    std::vector<cv::Point2f> scene_corners;
    cv::perspectiveTransform(model_corners, scene_corners, H);
    cv::Rect boundingBox = cv::boundingRect(scene_corners);

    cv::rectangle(imgQuery, boundingBox, cv::Scalar(0,255,0), 2); // Disegna un rettangolo
    cv::imshow("imgQuery", imgQuery);
    cv::waitKey(0);

    return Label(object_type, boundingBox);
}

