#include "../../include/ObjectDetector/ObjectDetector.h"
#include "../../include/Utils.h"

ObjectDetector::~ObjectDetector() {}

const size_t ObjectDetector::detect_object_whole_dataset(const Dataset& dataset, std::vector<std::vector<Label>>& out_labels){
    
    out_labels.clear();
    
    const std::vector<std::pair<std::vector<Label>, std::string>>& test_data = dataset.get_items();

    for(auto it=test_data.begin(); it != test_data.end(); ++it){
        
        std::map<std::string, std::vector<Label>> predicted_items;
        this->detect_objects_dataset(it->second, dataset, predicted_items);

        out_labels.push_back(predicted_items[it->second]); // per me si deve cambiare e ritornare std::map<std::string, std::vector<Label>> predicted_items;
    };

    return out_labels.size();
}