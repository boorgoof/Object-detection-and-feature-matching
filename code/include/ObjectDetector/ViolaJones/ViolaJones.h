//Gianluca Caregnato

#ifndef VIOLA_JONES_H
#define VIOLA_JONES_H

#include "../ObjectDetector.h"
#include "../../Label.h"
#include "../../Dataset.h"
#include "../../CustomErrors.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

/**
 * @brief ViolaJones class to detect objects in images using the Viola-Jones algorithm.
 *        The class derives from the abstract class ObjectDetector.
 */
class ViolaJones : public ObjectDetector {
private:
    /**
     * @brief The type of object to be detected.
     */
    Object_Type::Type type;
    /**
     * @brief The cascade classifier used for object detection.
     */
    cv::CascadeClassifier cascade;
    /**
     * @brief A map to associate object types with their corresponding cascade file paths.
     */
    static inline const std::map<Object_Type, std::string> objectTypeMap = {
        { Object_Type::Type::SUGAR_BOX, "../Image_generated/cascade_sugar_box_gray40/cascade.xml" },
        { Object_Type::Type::MUSTARD_BOTTLE, "../Image_generated/cascade_mustard_bottle_gray60/cascade.xml" },
        { Object_Type::Type::POWER_DRILL, "../Image_generated/cascade_power_drill_gray60/cascade.xml" }
    };

public:
    /**
     * @brief Constructor for the ViolaJones class.
     * @param type The type of object to be detected.
     */
    ViolaJones(const Object_Type& type);
    /**
     * @brief Destructor for the ViolaJones class.
     */
    ~ViolaJones();
    /**
     * @brief Deleted default constructor to prevent instantiation without parameters.
     */
    ViolaJones() = delete;
    

    /**
     * @brief Override of the detect_objects method to detect objects in a scene image.
     * @param src_img The scene image to detect objects from.
     * @param out_labels The output vector of detected labels.
     */
    void detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) override;
};

#endif