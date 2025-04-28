#include "../../../include/ObjectDetector/ViolaJones/ImageDatasetGenerator.h"
#include "../include/CustomErrors.h"

void ImageDatasetGenerator::generateRotatedImages(const std::string& inputImage, const std::string& inputMask, const std::string& outputFolder, int numImages, bool genLightVariations, int numLightVariations) {
    namespace fs = std::filesystem;
    double angleStep = 360.0 / numImages;

    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }

    cv::Mat imm = cv::imread(inputImage);
    if (imm.empty()) {
        throw CustomErrors::ImageLoadError("Error loading the image: ", inputImage);
    }
    cv::Mat mask = cv::imread(inputMask);
    if (mask.empty()) {
        throw CustomErrors::ImageLoadError("Error loading the mask: ", inputMask);
    }

    cv::Point2f center(imm.cols / 2.0f, imm.rows / 2.0f);
    fs::path inputPathImage(inputImage);
    fs::path inputPathMask(inputMask);

    std::string stem = inputPathImage.stem().string();
    std::string base;
    size_t pos = stem.find_last_of('_');
    base = stem.substr(0, pos);

    for (int i = 0; i < numImages; i++) {
        double angle = i * angleStep;
        cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat outputImm, outputMask;
        cv::warpAffine(imm, outputImm, rotMat, imm.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        cv::warpAffine(mask, outputMask, rotMat, mask.size());
        
        std::string outputImmname = base + "_rot_" + std::to_string(static_cast<int>(angle)) + "_color" + inputPathImage.extension().string();
        std::string outputMaskname = base + "_rot_" + std::to_string(static_cast<int>(angle)) + "_mask" + inputPathMask.extension().string();
        cv::imwrite((fs::path(outputFolder) / outputImmname).string(), outputImm);
        cv::imwrite((fs::path(outputFolder) / outputMaskname).string(), outputMask);

        if (genLightVariations)
            generateLightVariations((fs::path(outputFolder) / outputImmname).string(), (fs::path(outputFolder) / outputMaskname).string(), outputFolder, numLightVariations);
    }
}

void ImageDatasetGenerator::generateLightVariations(const std::string& inputImage, const std::string& inputMask, const std::string& outputFolder, int numImages) {
    namespace fs = std::filesystem;

    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }

    cv::Mat imm = cv::imread(inputImage);
    cv::Mat mask = cv::imread(inputMask);
    if (imm.empty()) {
        throw CustomErrors::ImageLoadError("Error loading the image: ", inputImage);
    }
    if (mask.empty()) {
        throw CustomErrors::ImageLoadError("Error loading the mask: ", inputMask);
    }

    int brightnessRange = 80;
    int brightnessStart = -brightnessRange / 2;
    int brightnessStep = brightnessRange / (numImages - 1);

    fs::path inputPath(inputImage);
    std::string stem = inputPath.stem().string();
    std::string base;
    size_t pos = stem.find_last_of('_');
    base = stem.substr(0, pos);

    int currentBrightness = brightnessStart;
    for (int i = 0; i < numImages; i++) {
        cv::Mat modifiedImage;
        imm.convertTo(modifiedImage, -1, 1, currentBrightness);

        
        std::string outputNameImm = base + "_light_" + std::to_string(currentBrightness) + "_color" + inputPath.extension().string();
        std::string outputNameMask = base + "_light_" + std::to_string(currentBrightness) + "_mask" + inputPath.extension().string();
        cv::imwrite((fs::path(outputFolder) / outputNameImm).string(), modifiedImage);
        cv::imwrite((fs::path(outputFolder) / outputNameMask).string(), mask);
        currentBrightness += brightnessStep;
    }
    if (fs::exists(inputImage)) {
        fs::remove(inputImage);
    }
    if (fs::exists(inputMask)) {
        fs::remove(inputMask);
    }
}

void ImageDatasetGenerator::generateGrayImage(const std::string& inputImage, const std::string& inputMask, const std::string& outputFolder){
    namespace fs = std::filesystem;

    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }

    cv::Mat imm = cv::imread(inputImage);
    cv::Mat mask = cv::imread(inputMask);
    if (imm.empty()) {
        throw CustomErrors::ImageLoadError("Error loading the image: ", inputImage);
    }
    if (mask.empty()) {
        throw CustomErrors::ImageLoadError("Error loading the mask: ", inputMask);
    }

    cv::Mat grayImm;
    cv::cvtColor(imm, grayImm, cv::COLOR_BGR2GRAY);
    cv::Mat blurred_gray;
    cv::GaussianBlur(grayImm, blurred_gray, cv::Size(3, 3), 0);

    fs::path inputPath(inputImage);
    std::string stem = inputPath.stem().string();
    std::string base;
    size_t pos = stem.find_last_of('_');
    base = stem.substr(0, pos);

    std::string outputNameImm = base + "_gray_color" + inputPath.extension().string();
    std::string outputNameMask = base + "_gray_mask" + inputPath.extension().string();
    
    cv::imwrite((fs::path(outputFolder) / outputNameImm).string(), grayImm);
    cv::imwrite((fs::path(outputFolder) / outputNameMask).string(), mask);
}