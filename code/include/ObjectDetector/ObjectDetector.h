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
    ObjectDetector() = default;
    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector(ObjectDetector&&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;
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

    const std::string& get_method() const { return this->method; }
    void set_method(const std::string& method) { this->method = method; }
    const std::string& get_filter1() const { return this->filter1; }
    void set_filter1(const std::string& filter1) { this->filter1 = filter1; }
    const std::string& get_filter2() const { return this->filter2; }
    void set_filter2(const std::string& filter2) { this->filter2 = filter2; }
    private:
    std::string method;
    std::string filter1;
    std::string filter2;
};

#endif // OBJECT_DETECTOR_H
