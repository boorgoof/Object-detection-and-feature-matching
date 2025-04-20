#include "../../include/ObjectDetector/ObjectDetector.h"
#include "../../include/Utils.h"

ObjectDetector::~ObjectDetector() {}

const size_t ObjectDetector::detect_object_whole_dataset(const Dataset& dataset, std::vector<std::vector<Label>>& out_labels){
    
    out_labels.clear();
    
    const std::vector<std::pair<std::vector<Label>, std::string>>& test_data = dataset.get_items();

    for(auto it=test_data.begin(); it != test_data.end(); ++it){
        const cv::Mat img = Utils::Loader::load_image(it->second);

        std::vector<Label> img_detections;
        this->detect_objects(img, dataset, img_detections);

        out_labels.push_back(img_detections);
    };

    return out_labels.size();
}