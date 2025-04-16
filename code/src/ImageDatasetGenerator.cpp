#include "../include/ImageDatasetGenerator.h"
#include "../include/CustomErrors.h"

void ImageDatasetGenerator::generateRotatedImages(const std::string& inputImage, const std::string& inputMask, const std::string& outputFolder, int numImages) {
    namespace fs = std::filesystem;
    double angleStep = 360.0 / numImages;

    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }

    cv::Mat imm = cv::imread(inputImage);
    if (imm.empty()) {
        throw CustomErrors::ImageLoadError("Error loading the image: " , inputImage);
    }
    cv::Mat mask = cv::imread(inputMask);
    if (mask.empty()) {
        throw CustomErrors::ImageLoadError("Error loading the mask: " , inputMask);
    }

    cv::Point2f center(imm.cols / 2, imm.rows / 2);
    fs::path inputPathImage(inputImage);
    fs::path inputPathMask(inputMask);
    for (int i = 0; i < numImages; i++) {
        double angle = i * angleStep;
        cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat outputImm, outputMask;
        cv::warpAffine(imm, outputImm, rotMat, imm.size());
        cv::warpAffine(imm, outputMask, rotMat, mask.size());
        std::string outputImmname = inputPathImage.stem().string() + "_rot_" + std::to_string(static_cast<int>(angle)) + inputPathImage.extension().string();
        cv::imwrite((fs::path(outputFolder) / outputImmname).string(), outputImm);
        std::string outputMaskname = inputPathMask.stem().string() + "_rot_" + std::to_string(static_cast<int>(angle)) + inputPathMask.extension().string();
        cv::imwrite((fs::path(outputFolder) / outputMaskname).string(), outputMask);
    }
}
