//Gianluca Caregnato

#include "../../../include/ObjectDetector/ViolaJones/ViolaJonesTraining.h"


int ViolaJonesTraining::PositiveSamplesFiles(const std::string &inputFolder, const std::string &outputDir, const std::string &fileName) {
    std::vector<std::string> images_filenames, masks_filenames;
    int numFile = Utils::Directory::split_model_img_masks(inputFolder, images_filenames, masks_filenames);
    for(int i = 0; i < numFile; i++) {
        std::cout << "image: " << images_filenames[i] << " " << "mask: " << masks_filenames[i] << std::endl;
    }

    std::string outputFile = outputDir + "/" + fileName;
    std::ofstream ofs(outputFile);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << outputFile << std::endl;
        return -1;
    }


    for (size_t i = 0; i < numFile; i++) {

        cv::Mat mask = Utils::Loader::load_image(masks_filenames[i]);
        cv::Mat image = Utils::Loader::load_image(images_filenames[i]);

        cv::Rect rect = cv::boundingRect(mask);
        if (rect.empty()) {
            std::cerr << "No non-zero pixels found in mask " << masks_filenames[i] << std::endl;
            continue;
        }
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);

        ofs << images_filenames[i] << " "
            << rect.x << " " << rect.y << " "
            << rect.width << " " << rect.height << "\n";
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
            ofs << imageName << "\n";
        }
    }
    ofs.close();
    return numFile;
}

