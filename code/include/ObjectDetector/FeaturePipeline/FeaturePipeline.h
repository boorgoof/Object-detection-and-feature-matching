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
            const std::vector<cv::KeyPoint>& query_keypoint,
            const std::vector<cv::KeyPoint>& model_keypoint,
            const cv::Mat& imgModel,
            const cv::Mat& maskModel,
            const cv::Mat& imgQuery,
            Object_Type object_type) const;
        
        /*
        Label findBoundingBox_2(const std::vector<cv::DMatch>& matches,
            const std::vector<cv::KeyPoint>& query_keypoint,
            const std::vector<cv::KeyPoint>& model_keypoint,
            const cv::Mat& imgModel,
            const cv::Mat& maskModel,
            const cv::Mat& imgQuery,
            Object_Type object_type) const;

        */
};

#endif // FEATUREPIPELINE_H
  /*
   DA eliminare dopo:

    1)
    SI USA Feature2D.detectAndCompute
    SI OTTIENE: std::vector<cv::KeyPoint> kp1, kp2; keypoints da immagine 1 e 2 
    
    nota:
    KeyPoint
    - float angle
    - intclass_id :object class (if the keypoints need to be clustered by an object they belong to) 
    - int octave: octave (pyramid layer) from which the keypoint has been extracted M
    - Point2f pt :coordinates of the keypoints (importante per noi)
    - float response: the response by which the most strong keypoints have been selected. 
    - float size diameter of the meaningful keypoint neighborhood 

    2) 
    SI USA DescriptorMatcher.match
    SI OTTIENE  std::vector<cv::DMatch> matches;            (solitamente si prendono i migliori) 
   
    nota:
    struct DMatch {
        int queryIdx;  indice del descrittore/keypoint nella "query" (immagine di test)
        int trainIdx;  indice del descrittore/keypoint nella "train" (model)
        float distance; distanza tra i descrittori: quindi piu piccola è meglio è.
    };

    3) 
    I MATCH VENGONO TRASFOMATI IN PUNTI
    std::vector<cv::Point2f> src_pts, dst_pts;      punti corrispondenti ai match (sono le coordinate)

    src_pts.push_back(kp1[goodMatches[i].queryIdx].pt);
    dst_pts.push_back(kp2[goodMatches[i].trainIdx].pt);

    */

