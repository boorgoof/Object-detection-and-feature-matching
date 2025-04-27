#ifndef FEATUREPIPELINE_H
#define FEATUREPIPELINE_H

#include "../ObjectDetector.h"
#include "FeatureDetector.h"
#include "FeatureMatcher.h"
#include <opencv2/opencv.hpp>
#include "../../Label.h"
#include "Features.h"

class ImageFilter;

/**
 * @brief FeaturePipeline class to detect objects in images using feature detection and matching.
 *        This class implements the ObjectDetector interface.
 * 
 */
class FeaturePipeline : public ObjectDetector {

    private:
       /**
        * @brief FeatureDetector pointer to the feature detector used by the pipeline.
        */
        FeatureDetector* detector;

        /**
         * @brief FeatureMatcher pointer to the feature matcher used by the pipeline.
         */
        FeatureMatcher* matcher;

        /**
         * @brief ImageFilter pointer to the image filter used by the pipeline.
         */
        ImageFilter* model_imagefilter;
        /**
         * @brief ImageFilter pointer to the image filter used by the pipeline.
         */
        ImageFilter* test_imagefilter;

        /**
         * @brief Dataset reference to the dataset used by the pipeline.
         */
        Dataset& dataset;
        std::vector<ModelFeatures> models_features;

        /**
         * @brief method to initialize all the models' features.
         */
        void init_models_features();

        /**
         * @brief method to check and to update the compatibility between the detector and matcher.
         */
        void update_detector_matcher_compatibility();

    public:

        FeaturePipeline(FeatureDetector* fd, FeatureMatcher* fm, Dataset& dataset, ImageFilter* model_imagefilter = nullptr, ImageFilter* test_imagefilter = nullptr)
            : detector{fd}, matcher{fm}, dataset{dataset}, ObjectDetector{DetectorType::toString(fd->getType())+"-"+MatcherType::toString(fm->getType())} { this->model_imagefilter = model_imagefilter; this->test_imagefilter = test_imagefilter; this->update_detector_matcher_compatibility(); this->init_models_features();}
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
    
        /**
         * @brief method to detect objects in the scene image.
         * @param src_img the scene (test) image
         * @param out_labels the output vector of labels that will contain the detected objects
         */
        void detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) override;

        /**
        * @brief method to find the label (so the bounding box) of the object in the scene image.
        * @param matches the matches between the model and the scene
        * @param model_keypoint the keypoints of the model
        * @param scene_keypoint the keypoints of the scene (test image)
        * @param img_model the model image
        * @param mask_model the model mask
        * @param img_scene the scene (test) image
        * @param object_type the object type to find
        * @return the Label that containt the bounding box of the object in the scene
        */
        Label findBoundingBox(const std::vector<cv::DMatch>& matches,
            const std::vector<cv::KeyPoint>& model_keypoint,
            const std::vector<cv::KeyPoint>& scene_keypoint,
            const cv::Mat& img_model,
            const cv::Mat& mask_model,
            const cv::Mat& img_scene,
            Object_Type object_type) const ;
        
       
};

#endif // FEATUREPIPELINE_H


