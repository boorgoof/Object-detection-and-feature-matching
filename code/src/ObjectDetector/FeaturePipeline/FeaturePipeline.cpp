#include "../../../include/ObjectDetector/FeaturePipeline/FeaturePipeline.h"

/*
1) trovi le feature di tutte le immagini in models
2) per ogni immagine nel teste le confornti con quelle in models
3) si prende il modello con piu match
4) se il match viene considerato buono, viene localizzato in un bounding box

*/

const size_t FeaturePipeline::detect_objects(const cv::Mat& src_img, Object_Type object_type, std::vector<Label>& out_labels){
    //add functionality using class members 
    //cv::Ptr<cv::Feature2D> feature_detector
    //cv::Ptr<cv::DescriptorMatcher> feature_matcher
    
    std::vector<cv::KeyPoint> src_keypoints;
    cv::Mat src_desc;
    this->detectFeatures(src_img,  cv::Mat(),  src_keypoints, src_desc); // in teoria con cv::Mat() non applica una maschera
    //ritorna i keypoints e i descrittori dell'immagine di input

    //feature matching passando keypoints e descrittori

    std::pair<ModelFeatures, std::vector<cv::DMatch>> bestModel = selectBestModel(src_desc);
  
    ;
    //-- Draw matches
    //cv::Mat img_matches;
    //cv::drawMatches(src_img, src_keypoints, bestModel.first.image, bestModel.first.keypoints, bestModel.second, img_matches );
    //cv::imshow("Matches", img_matches);
    //cv::waitKey(0);

    out_labels.push_back(this->findBoundingBox(bestModel.second, 
        src_keypoints, 
        bestModel.first.keypoints, 
        bestModel.first.image,
        bestModel.first.mask,
        src_img,
        object_type));

    return 0;
}


std::pair<ModelFeatures, std::vector<cv::DMatch>> FeaturePipeline::selectBestModel(const cv::Mat& query_descriptors) const {
    
    int best_model_idx = 0;
    int best_score = 0;
    std::vector<cv::DMatch> good_matches;

    for (size_t i = 0; i < models_features.size(); ++i) {
        good_matches.clear();

        const auto& model = models_features[i];
        if (model.descriptors.empty()) 
            std::cout << "Errore models_features is not initialized correctly";

        std::vector<cv::DMatch> matches;
        this->matchFeatures(query_descriptors, model.descriptors, matches);
        
        std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {return a.distance < b.distance;});

        
        for (size_t j = 0; j < matches.size(); ++j) {
            //std::cout << matches[j].distance << std::endl;
            if (matches[j].distance < 130.0f) {
                good_matches.push_back(matches[j]);
            }
        }
        //std::cout << good_matches.size() << std::endl;

        int score = good_matches.size();

        if (score > best_score) {
            //std::cout << "changing score and index" << std::endl;
            best_score = score;
            best_model_idx = i;
            //std::cout << "best model index: " << best_model_idx << std::endl;
            //std::cout << "best score: " << best_score << std::endl;
        }
        //std::cout << "i: " << i << std::endl;
    }
    std::cout << best_model_idx << std::endl;
    std::cout << best_score << std::endl;


    return { models_features[best_model_idx], good_matches };
}


void FeaturePipeline::setModelsfeatures(const Dataset dataset) {
    
    const std::vector<std::pair<std::string, std::string>>& model_pairs = dataset.get_models();

    for (const auto& model_pair : model_pairs) {

        cv::Mat img = cv::imread(model_pair.first, cv::IMREAD_COLOR);
        cv::Mat mask = cv::imread(model_pair.second, cv::IMREAD_GRAYSCALE);

        if (img.empty() || mask.empty()){
            std::cout<<"errore train o mask non caricati";
        }

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        this->detectFeatures(img, mask, keypoints, descriptors);

        ModelFeatures model{
            dataset.get_type(),           
            model_pair.first,             
            img,
            mask,
            keypoints,
            descriptors
        };

        this->models_features.push_back(model);

    }

}


Label FeaturePipeline::findBoundingBox(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& query_keypoint,
    const std::vector<cv::KeyPoint>& model_keypoint,
    const cv::Mat& imgModel,
    const cv::Mat& maskModel,
    const cv::Mat& imgQuery,
    Object_Type object_type) const 
{
    const int minMatches= 10;

    std::cout << "matches: " << matches.size() << std::endl;

    if (matches.size() < minMatches) {
        std::cerr << "not enough matches " << matches.size() << ". Min: " << minMatches << std::endl;
        return Label(object_type, cv::Rect());
    }

    std::vector<cv::Point2f> query_pts, model_pts;

    for (const auto& match : matches) {
        query_pts.push_back(query_keypoint[match.queryIdx].pt);
        model_pts.push_back(model_keypoint[match.trainIdx].pt);
    }

    //cv::Mat mask;
    cv::Mat H = cv::findHomography(model_pts, query_pts, cv::RANSAC, 5.0);
    if (H.empty())
        return Label(object_type, cv::Rect());      //throw error and catch it

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



void FeaturePipeline::detectFeatures(const cv::Mat& img, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {
    feature_detector->detectAndCompute(img, mask, keypoints, descriptors);
}

void FeaturePipeline::matchFeatures(const cv::Mat& queryDescriptors,const cv::Mat& trainDescriptors, std::vector<cv::DMatch>& matches) const {
    feature_matcher->match(queryDescriptors, trainDescriptors, matches);
}

void FeaturePipeline::setFeatureDetector(const cv::Ptr<cv::Feature2D>& features_detector) {
    this->feature_detector = features_detector;
}

void FeaturePipeline::setFeatureMatcher(const cv::Ptr<cv::DescriptorMatcher>& features_matcher) {
    this->feature_matcher = features_matcher;
}