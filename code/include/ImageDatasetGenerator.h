#ifndef ImageDatasetGenerator_h
#define ImageDatasetGenerator_h

#include <opencv2/opencv.hpp>
#include <filesystem>

class ImageDatasetGenerator{
    public:
    void generateRotatedImages(const std::string&,  const std::string&, const std::string&, int, bool, int);
    void generateLightVariations(const std::string&, const std::string&, int);
};


#endif