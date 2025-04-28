#ifndef ImageDatasetGenerator_h
#define ImageDatasetGenerator_h

#include <opencv2/opencv.hpp>
#include <filesystem>

class ImageDatasetGenerator{
    public:
    /*
    @brief Generates a dataset of images with various rotation.
    @param inputImage The path to the input image.
    @param inputMask The path to the input mask.
    @param outputFolder The folder where the generated images will be saved.
    @param numImages The number of images to generate with different rotation.
    @param genLightVariations Whether to generate light variations.
    @param numLightVariations The number of light variations to generate.
    */
    void generateRotatedImages(const std::string& inputImage, const std::string& inputMask, const std::string& outputFolder, int numImages, bool genLightVariations, int numLightVariations);
    /*
    @brief Generates a dataset of images with various light variations.
    @param inputImage The path to the input image.
    @param inputMask The path to the input mask.
    @param outputFolder The folder where the generated images will be saved.
    @param numImages The number of images to generate with different light variations.
    */
    void generateLightVariations(const std::string& inputImage, const std::string& inputMask, const std::string& outputFolder, int numImages);
    /*
    @brief Generates a dataset of images with gray variations.
    @param inputImage The path to the input image.
    @param inputMask The path to the input mask.
    @param outputFolder The folder where the generated images will be saved.
    */
    void generateGrayImage(const std::string& inputImage, const std::string& inputMask, const std::string& outputFolder);

};


#endif