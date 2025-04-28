#include "../include/ObjectDetector/ViolaJones/ImageDatasetGenerator.h"
#include "../include/Utils.h"
#include "../include/ObjectDetector/ViolaJones/ViolaJonesTraining.h"

int main(){

    // Generate rotated images for Power_drill
    ImageDatasetGenerator generator;
    
    std::string inputFolder = "../dataset/035_power_drill/models";
    std::vector<std::string> images_filenames, masks_filenames;
    int numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);

    for(int i = 0; i < numFile; i++){
        generator.generateRotatedImages(images_filenames[i], masks_filenames[i], "../Image_generated/Power_drill_Generated", 10, true, 6);
    }
    
    // Generate rotated images for Mustard_bottle
    inputFolder = "../dataset/006_mustard_bottle/models";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);
    for(int i = 0; i < numFile; i++){
        generator.generateRotatedImages(images_filenames[i], masks_filenames[i], "../Image_generated/Mustard_bottle_Generated", 10, true, 6);
    }
    
    // Generate rotated images for Sugar_box
    inputFolder = "../dataset/004_sugar_box/models";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);
    for(int i = 0; i < numFile; i++){
        generator.generateRotatedImages(images_filenames[i], masks_filenames[i], "../Image_generated/Sugar_box_Generated", 10, true, 6);
    }

    // Generate grayscale images for Power_drill
    inputFolder = "../Image_generated/Power_drill_Generated";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);
    for(int i = 0; i < numFile; i++){
        generator.generateGrayImage(images_filenames[i], masks_filenames[i], "../Image_generated/Power_drill_GeneratedGray");
    }
    
    // Generate grayscale images for Mustard_bottle
    inputFolder = "../Image_generated/Mustard_bottle_Generated";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);
    for(int i = 0; i < numFile; i++){
        generator.generateGrayImage(images_filenames[i], masks_filenames[i], "../Image_generated/Mustard_bottle_GeneratedGray");
    }
    
    // Generate grayscale images for Sugar_box
    inputFolder = "../Image_generated/Sugar_box_Generated";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);
    for(int i = 0; i < numFile; i++){
        generator.generateGrayImage(images_filenames[i], masks_filenames[i], "../Image_generated/Sugar_box_GeneratedGray");
    }

    // Train Viola-Jones using generated samples
    ViolaJonesTraining training;
    std::vector<std::string> inputFolders = { "../Image_generated/Mustard_bottle_GeneratedGray", "../Image_generated/Power_drill_GeneratedGray", "../Image_generated/Sugar_box_GeneratedGray" };
    // Mustard_bottle samples
    std::string outputDir = "../Image_generated";
    std::string fileName = "Mustard_bottleG.txt";
    std::string fileNameNegative = "Mustard_bottle_negativeG.txt";
    int i = training.PositiveSamplesFiles(inputFolders[0], outputDir, fileName);
    int j = training.NegativeSamplesFiles({inputFolders[1], inputFolders[2]}, outputDir, fileNameNegative);

    std::cout << "Mustard_bottle Positive samples: " << i << std::endl;
    std::cout << "Mustard_bottle Negative samples: " << j << std::endl;

    // Power_drill samples
    fileName = "Power_drillG.txt";
    fileNameNegative = "Power_drill_negativeG.txt";
    i = training.PositiveSamplesFiles(inputFolders[1], outputDir, fileName);
    j = training.NegativeSamplesFiles({inputFolders[0], inputFolders[2]}, outputDir, fileNameNegative);

    std::cout << "Power_drill Positive samples: " << i << std::endl;
    std::cout << "Power_drill Negative samples: " << j << std::endl;

    // Sugar_box samples
    fileName = "Sugar_boxG.txt";
    fileNameNegative = "Sugar_box_negativeG.txt";
    i = training.PositiveSamplesFiles(inputFolders[2], outputDir, fileName);
    j = training.NegativeSamplesFiles({inputFolders[0], inputFolders[1]}, outputDir, fileNameNegative);

    std::cout << "Sugar_box Positive samples: " << i << std::endl;
    std::cout << "Sugar_box Negative samples: " << j << std::endl;
}