#include "../include/ImageDatasetGenerator.h"

int main(){
    std::cout << "Hello World!" << std::endl;
    ImageDatasetGenerator generator;
    std::string inputFolder = "004_sugar_box/models";
    std::string Image = "/home/gian-pc/Desktop/MiddleProject_ComputerVision/dataset/035_power_drill/models/view_0_000_color.png";
    std::string Mask = "/home/gian-pc/Desktop/MiddleProject_ComputerVision/dataset/035_power_drill/models/view_0_000_mask.png";
    std::string outputFolder = "/home/gian-pc/Desktop/MiddleProject_ComputerVision/Image_generated/Power_drill_Generated";

    generator.generateRotatedImages(Image,Mask ,outputFolder, 10, true, 10);

    //ImageDatasetGenerator::generateRotatedImages(const std::string& inputImage, const std::string& outputFolder, int numImages) 
}