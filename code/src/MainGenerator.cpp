#include "../include/ImageDatasetGenerator.h"
#include "../include/Utils.h"
#include "../include/ViolaJonesTraining.h"

int main(){/*
    ImageDatasetGenerator generator;
    
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

    inputFolder = "../Image_generated/Power_drill_Generated";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);

    for(int i = 0; i < numFile; i++){
        generator.generateGrayImage(images_filenames[i], masks_filenames[i], "../Image_generated/Power_drill_GeneratedGray");
    }
    inputFolder = "../Image_generated/Mustard_bottle_Generated";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);

    for(int i = 0; i < numFile; i++){
        generator.generateGrayImage(images_filenames[i], masks_filenames[i], "../Image_generated/Mustard_bottle_GeneratedGray");
    }
    
    inputFolder = "../Image_generated/Sugar_box_Generated";
    numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);
    for(int i = 0; i < numFile; i++){
        generator.generateGrayImage(images_filenames[i], masks_filenames[i], "../Image_generated/Sugar_box_GeneratedGray");
    }
*/
    
    ViolaJonesTraining training;
    std::vector<std::string> inputFolders = { "../Image_generated/Mustard_bottle_GeneratedGray", "../Image_generated/Power_drill_GeneratedGray", "../Image_generated/Sugar_box_GeneratedGray" };


    std::string outputDir = "../Image_generated";
    std::string fileName = "Mustard_bottleG.txt";
    std::string fileNameNegative = "Mustard_bottle_negativeG.txt";
    int i = training.PositiveSamplesFiles(inputFolders[0], outputDir, fileName);
    int j = training.NegativeSamplesFiles({inputFolders[1], inputFolders[2]}, outputDir, fileNameNegative);

    std::cout << "Mustard_bottle Positive samples: " << i << std::endl;
    std::cout << "Mustard_bottle Negative samples: " << j << std::endl;

    fileName = "Power_drillG.txt";
    fileNameNegative = "Power_drill_negativeG.txt";
    i = training.PositiveSamplesFiles(inputFolders[1], outputDir, fileName);
    j = training.NegativeSamplesFiles({inputFolders[0], inputFolders[2]}, outputDir, fileNameNegative);

    std::cout << "Power_drill Positive samples: " << i << std::endl;
    std::cout << "Power_drill Negative samples: " << j << std::endl;

    fileName = "Sugar_boxG.txt";
    fileNameNegative = "Sugar_box_negativeG.txt";
    i = training.PositiveSamplesFiles(inputFolders[2], outputDir, fileName);
    j = training.NegativeSamplesFiles({inputFolders[0], inputFolders[1]}, outputDir, fileNameNegative);

    std::cout << "Sugar_box Positive samples: " << i << std::endl;
    std::cout << "Sugar_box Negative samples: " << j << std::endl;
}