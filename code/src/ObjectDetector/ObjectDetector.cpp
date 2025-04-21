#include "../../include/ObjectDetector/ObjectDetector.h"
#include "../../include/Utils.h"

ObjectDetector::~ObjectDetector() {}

const size_t ObjectDetector::detect_object_whole_dataset(const Dataset& dataset, std::map<std::string, std::vector<Label>>& predicted_items){
    
    predicted_items.clear();
    
    const std::vector<std::pair<std::vector<Label>, std::string>>& test_data = dataset.get_items();

    for(auto it=test_data.begin(); it != test_data.end(); ++it){
        
        this->detect_objects_dataset(it->second, dataset, predicted_items);
    };

    return predicted_items.size();
}