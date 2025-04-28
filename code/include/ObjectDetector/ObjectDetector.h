#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "../Label.h"
#include "../Dataset.h"


/**
 * @brief ObjectDetector class to detect objects in images.
 *        This is an abstract class that defines the interface for all object detectors.
 */
class ObjectDetector{
    public:
    ObjectDetector() {model_filter_name = ""; test_filter_name = "";};
    /*
    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;
    */
    ObjectDetector(ObjectDetector&&) = delete;
    ObjectDetector& operator=(ObjectDetector&&) = delete;
    virtual ~ObjectDetector() = 0;

    /**
     * @brief method to detect objects in a scene image
     * @param src_img the scene image to detect objects from
     * @param out_labels the output vector of detected labels
     */
    virtual void detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) = 0;

    /**
     * @brief method to detect objects in a whole dataset: detect objects in each image of the dataset and stores the results in a map.
     * @param dataset the dataset to detect objects from . Remember a dataset manages a single object's test images
     * @param predicted_items the output map of predicted items
     * @return the number of detected objects
     */
    const size_t detect_object_whole_dataset(const Dataset& dataset, std::map<std::string, std::vector<Label>>& predicted_items);

    const std::string& get_method_name() const { return this->method_name; }
    void set_method_name(const std::string& method_name) { this->method_name = method_name; }
    const std::string& get_model_filter_name() const { return this->model_filter_name; }
    void set_model_filter_name(const std::string& model_filter_name) { this->model_filter_name = model_filter_name; }
    const std::string& get_test_filter_name() const { return this->test_filter_name; }
    void set_test_filter_name(const std::string& test_filter_name) { this->test_filter_name = test_filter_name; }
    private:
    std::string method_name;
    std::string model_filter_name;
    std::string test_filter_name;
};

#endif // OBJECT_DETECTOR_H
