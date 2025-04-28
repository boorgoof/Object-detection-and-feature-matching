#ifndef ViolaJonesTraining_h
#define ViolaJonesTraining_h

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <random>
#include "../../Utils.h"


class ViolaJonesTraining {
public:
    /**
     * @brief Generates a file with the set of positive samples for training the Viola-Jones classifier.
     * @param inputFolder Path to the folder containing the images and masks.
     * @param outputDir Path to save the generated file.
     * @param fileName Name of the output file.
     * @return Number of positive samples generated.
     */
    int PositiveSamplesFiles(const std::string &inputFolder, const std::string &outputDir, const std::string &fileName );
    /**
     * @brief Generates a file with the set of negative samples for training the Viola-Jones classifier.
     * @param inputDir Path to the folder containing the images.
     * @param outputDir Path to save the generated file.
     * @param fileName Name of the output file.
     * @return Number of negative samples generated.
     */
    int NegativeSamplesFiles(const std::vector<std::string> &inputDir, const std::string &outputDir, const std::string &fileName);
};

#endif