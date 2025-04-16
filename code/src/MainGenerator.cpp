#include "../include/ImageDatasetGenerator.h"

int main(){
    ImageDatasetGenerator generator;
    std::string inputFolder = "004_sugar_box/models";

    generator.generateRotatedImages("/home/gian-pc/Desktop/MiddleProject_ComputerVision/dataset/004_sugar_box/models/view_0_001_color.png","/home/gian-pc/Desktop/MiddleProject_ComputerVision/Image_generated/Mustard_bottle_generated", 10);

    //ImageDatasetGenerator::generateRotatedImages(const std::string& inputImage, const std::string& outputFolder, int numImages) 
}