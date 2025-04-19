#include "../include/ViolaJonesTraining.h"



int ViolaJonesTraining::PositiveSamplesFiles(const std::string &inputFolder, const std::string &outputDir, const std::string &fileName) {
    std::vector<std::string> images_filenames, masks_filenames;
    int numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);

    // Example rectangle coordinates
    int x = 130, y = 60, w = 360, h = 345;

    std::string outputFile = outputDir + "/" + fileName;
    std::ofstream ofs(outputFile);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << outputFile << std::endl;
        return -1;
    }
    for (const auto &imageName : images_filenames) {
        ofs << "positive/" << imageName << " " << x << " " << y << " " << w << " " << h << "\n";
    }
    ofs.close();

    return numFile;
}
int ViolaJonesTraining::NegativeSamplesFiles(const std::vector<std::string> &inputDir, const std::string &outputDir, const std::string &fileName) {
    
    std::string outputFile = outputDir + "/" + fileName;
    std::ofstream ofs(outputFile);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << outputFile << std::endl;
        return -1;
    }
    int fileIndex = 1;
    int numFile = 0;
    for (const auto &dir : inputDir) {
        std::vector<std::string> images_filenames, masks_filenames;
        numFile += Utils::Directory::split_model_img_masks(dir, images_filenames, masks_filenames);
        for (const auto &imageName : images_filenames) {
            ofs << "negative/" << imageName << "\n";
        }
    }
    ofs.close();
    return numFile;
}