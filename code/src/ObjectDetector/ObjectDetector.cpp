#include "../../include/ObjectDetector/ObjectDetector.h"
#include "../../include/Utils.h"

ObjectDetector::~ObjectDetector() {}

const size_t ObjectDetector::detect_object_whole_dataset(const Dataset& dataset, std::map<std::string, std::vector<Label>>& predicted_items){
    
    predicted_items.clear();
    
    const std::map<std::string, std::vector<Label>>& test_data = dataset.get_test_items();

    for(auto test_item : test_data){
        
        this->detect_objects_dataset(test_item.first, dataset, predicted_items);
    };

    return predicted_items.size();
}