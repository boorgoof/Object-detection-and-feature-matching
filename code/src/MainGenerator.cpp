#include "../include/ImageDatasetGenerator.h"
#include "../include/Utils.h"

int main(){
    std::cout << "Hello World!" << std::endl;
    ImageDatasetGenerator generator;
    
    /*std::string Image = "/home/gian-pc/Desktop/MiddleProject_ComputerVision/dataset/035_power_drill/models/view_0_000_color.png";
    std::string Mask = "/home/gian-pc/Desktop/MiddleProject_ComputerVision/dataset/035_power_drill/models/view_0_000_mask.png";
    std::string outputFolder = "/home/gian-pc/Desktop/MiddleProject_ComputerVision/Image_generated/Power_drill_Generated";

    generator.generateRotatedImages(Image,Mask ,outputFolder, 10, true, 10);*/

    //ImageDatasetGenerator::generateRotatedImages(const std::string& inputImage, const std::string& outputFolder, int numImages)
    std::string inputFolder = "../dataset/035_power_drill/models";
    std::vector<std::string> images_filenames, masks_filenames;
    int numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);

    for(int i = 0; i < numFile; i++){
        generator.generateRotatedImages(images_filenames[i], masks_filenames[i], "../Image_generated/Power_drill_Generated", 10, true, 6);
    }
    inputFolder = "../dataset/006_mustard_bottle/models";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);

    for(int i = 0; i < numFile; i++){
        generator.generateRotatedImages(images_filenames[i], masks_filenames[i], "../Image_generated/Mustard_bottle_Generated", 10, true, 6);
    }
    
    inputFolder = "../dataset/004_sugar_box/models";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);
    for(int i = 0; i < numFile; i++){
        generator.generateRotatedImages(images_filenames[i], masks_filenames[i], "../Image_generated/Sugar_box_Generated", 10, true, 6);
    }


}