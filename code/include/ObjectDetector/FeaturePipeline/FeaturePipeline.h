#ifndef FEATUREPIPELINE_H
#define FEATUREPIPELINE_H


#include "../ObjectDetector.h"
#include "FeatureStrategy.h"
#include <opencv2/opencv.hpp>


class FeaturePipeline : public ObjectDetector {

    private:
        std::unique_ptr<FeatureStrategy> strategy;
    
    public:
        FeaturePipeline(std::unique_ptr<FeatureStrategy>&& strategy) // chiedere bene a matte
            : strategy(std::move(strategy)) {}
    
        void setStrategy(std::unique_ptr<FeatureStrategy>&& new_strategy) {
            strategy = std::move(new_strategy);
        }
    
        void detect_objects(const cv::Mat& src_img, const Dataset& dataset, std::vector<Label>& out_labels) override ;

        Label findBoundingBox_1(const std::vector<cv::DMatch>& matches,
            const std::vector<cv::KeyPoint>& query_keypoint,
            const std::vector<cv::KeyPoint>& model_keypoint,
            const cv::Mat& imgModel,
            const cv::Mat& maskModel,
            const cv::Mat& imgQuery,
            Object_Type object_type) const;

        
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

