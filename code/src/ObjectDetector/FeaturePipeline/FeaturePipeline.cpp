#include "../../../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"
#include "../../../include/ObjectDetector/FeaturePipeline/ImageFilter.h"

void FeaturePipeline::init_models_features() {
    this->models_features.clear();
    this->detector->detectModelsFeatures(this->dataset.get_models(), this->models_features, this->model_imagefilter);
}

void FeaturePipeline::update_detector_matcher_compatibility() {
    if (this->detector->getType() == DetectorType::Type::ORB && this->matcher->getType() == MatcherType::Type::FLANN) {
        delete this->matcher;
        this->matcher = new FeatureMatcher(MatcherType::Type::FLANN, new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)));
    }
}

FeaturePipeline::~FeaturePipeline() {
    delete this->detector;
    delete this->matcher;
    delete this->model_imagefilter;
    delete this->test_imagefilter;
}

void FeaturePipeline::detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) {

    out_labels.clear();

    //models' features are already detected and stored in the pipeline (they always remain the same for every test image, so they are detected only once)

    cv::Mat src_img_filtered;
    //test image filtering if the filter component is present
    if (this->test_imagefilter != nullptr) {
        src_img_filtered = this->test_imagefilter->apply_filters(src_img);
    }


    //detects test image features
    QueryFeatures query_features;
    this->detector->detectFeatures(src_img_filtered, query_features.keypoints, query_features.descriptors);
    
    //matches test image features with every models' features and store them in out_matches
    std::vector<std::vector<cv::DMatch>> out_matches;
    for(ModelFeatures model_features : this->models_features){
        std::vector<cv::DMatch> out_matches_t;
        this->matcher->matchFeatures(query_features.descriptors, model_features.descriptors, out_matches_t);
        out_matches.push_back(out_matches_t);
    }

    //finds the best model (the one with the most matches)
    int best_model_idx = -1;
    size_t best_score = 0;
    for (size_t i = 0; i < out_matches.size(); ++i) {
        if (out_matches[i].size() > best_score) {
            best_score = out_matches[i].size();
            best_model_idx = static_cast<int>(i);
        }
    }

    if(best_model_idx == -1){
        std::cout << "detect object: Nessun modello trovato" << std::endl; // eliminare solo il cout
        return;
    }
    //calculates bounding box of the object found in the test image
    cv::Mat imgModel = Utils::Loader::load_image(this->dataset.get_models()[best_model_idx].first);
    cv::Mat maskModel = Utils::Loader::load_image(this->dataset.get_models()[best_model_idx].second);
    Label labelObj = findBoundingBox(out_matches[best_model_idx], query_features.keypoints, this->models_features[best_model_idx].keypoints, imgModel,  maskModel, src_img_filtered, this->dataset.get_type());
    
    out_labels.push_back(labelObj);
}
/*
Label FeaturePipeline::findBoundingBox(

    void FeaturePipeline::detect_objects(const cv::Mat &src_img, std::vector<Label> &out_labels) {
    } const std::vector<cv::DMatch> &matches,
    const std::vector<cv::KeyPoint> &query_keypoint,
    const std::vector<cv::KeyPoint> &model_keypoint,
    const cv::Mat &imgModel,
    const cv::Mat &maskModel,
    const cv::Mat &imgQuery,
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
*/

//METODO OPENCV MA NON VA - e invece sì ;)
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