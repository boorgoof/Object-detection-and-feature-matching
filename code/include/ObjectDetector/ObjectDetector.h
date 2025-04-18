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

    virtual const size_t detect_objects(const cv::Mat& src_img, Object_Type object_type, std::vector<Label>& out_labels) = 0;
    const size_t detect_object_whole_dataset(const Dataset& dataset, std::vector<std::vector<Label>>& out_labels);
};