#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "../Label.h"
#include "../Dataset.h"

class ObjectDetector{
    public:
    ObjectDetector(const std::string& method) : method(method) {};
    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector(ObjectDetector&&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;
    ObjectDetector& operator=(ObjectDetector&&) = delete;
    virtual ~ObjectDetector() = 0;

    virtual void detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) = 0;
    const size_t detect_object_whole_dataset(const Dataset& dataset, std::map<std::string, std::vector<Label>>& predicted_items);

    const std::string& get_method() const { return this->method; }
    private:
    std::string method;
};

#endif // OBJECT_DETECTOR_H
