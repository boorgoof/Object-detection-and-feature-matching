//Federico Meneghetti

#include "../../include/ObjectDetector/ObjectDetector.h"
#include "../../include/Utils.h"

ObjectDetector::~ObjectDetector() {}

const size_t ObjectDetector::detect_object_whole_dataset(const Dataset& dataset, std::map<std::string, std::vector<Label>>& predicted_items){
    
    predicted_items.clear();
    
    const std::map<std::string, std::vector<Label>>& test_data = dataset.get_test_items();

    for(auto test_item : test_data){
        std::cout << "\t\timage " << test_item.first << std::endl;
        this->detect_objects(Utils::Loader::load_image(test_item.first), predicted_items[test_item.first]);
    };

    return predicted_items.size();
}