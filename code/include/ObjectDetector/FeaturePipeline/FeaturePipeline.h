#ifndef FEATUREPIPELINE_H
#define FEATUREPIPELINE_H


#include "../ObjectDetector.h"
#include "FeatureDetector.h"
#include "FeatureMatcher.h"
#include <opencv2/opencv.hpp>
#include "../../Label.h"
#include "Features.h"
class ImageFilter;

class FeaturePipeline : public ObjectDetector {

    private:
        FeatureDetector* detector;
        FeatureMatcher* matcher;
    
        ImageFilter* model_imagefilter;
        ImageFilter* test_imagefilter;

        Dataset& dataset;
        std::vector<ModelFeatures> models_features;

        void init_models_features();
        void update_detector_matcher_compatibility();

    public:
        FeaturePipeline(FeatureDetector* fd, FeatureMatcher* fm, Dataset& dataset, ImageFilter* model_imagefilter = nullptr, ImageFilter* test_imagefilter = nullptr)
            : detector{fd}, matcher{fm}, dataset{dataset} {this->model_imagefilter = model_imagefilter; this->test_imagefilter = test_imagefilter; this->update_detector_matcher_compatibility(); this->init_models_features();}
        ~FeaturePipeline();

        void addDetectorComponent(FeatureDetector* fd) {
            detector = fd;
        }
        void addMatcherComponent(FeatureMatcher* fm) {
            matcher = fm;
        }
        void addModelImageFilterComponent(ImageFilter* imagefilter) {
            this->model_imagefilter = imagefilter;
        }
        void addTestImageFilterComponent(ImageFilter* imagefilter) {
            this->test_imagefilter = imagefilter;
        }

        void setDataset(Dataset& dataset) {
            this->dataset = dataset;
        }
        const Dataset& getDataset() const {
            return this->dataset;
        }
    
        void detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) override;

        Label findBoundingBox(const std::vector<cv::DMatch>& matches,
            const std::vector<cv::KeyPoint>& model_keypoint,
            const std::vector<cv::KeyPoint>& scene_keypoint,
            const cv::Mat& img_model,
            const cv::Mat& mask_model,
            const cv::Mat& img_scene,
            Object_Type object_type) const ;
        
       
};

#endif // FEATUREPIPELINE_H


