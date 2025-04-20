#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "../Label.h"
#include "../Dataset.h"

class ObjectDetector{
    public:
    ObjectDetector() = default;
    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector(ObjectDetector&&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;
    ObjectDetector& operator=(ObjectDetector&&) = delete;
    virtual ~ObjectDetector() = 0;

    virtual void detect_objects(const cv::Mat& src_img, const Dataset& dataset, std::vector<Label>& out_labels) = 0;
    const size_t detect_object_whole_dataset(const Dataset& dataset, std::vector<std::vector<Label>>& out_labels);
};

#endif // OBJECT_DETECTOR_H
