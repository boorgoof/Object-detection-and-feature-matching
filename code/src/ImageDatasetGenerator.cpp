#include "../include/ImageDatasetGenerator.h"
#include "../include/CustomErrors.h"

void ImageDatasetGenerator::generateRotatedImages(const std::string& inputImage, const std::string& outputFolder, int numImages) {
    namespace fs = std::filesystem;
    double angleStep = 360.0 / numImages;

    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }

    cv::Mat src = cv::imread(inputImage);
    if (src.empty()) {
        throw CustomErrors::ImageLoadError("Error loading image: " , inputImage);
    }

    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    fs::path inputPath(inputImage);
    for (int i = 0; i < numImages; i++) {
        double angle = i * angleStep;
        cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat dst;
        cv::warpAffine(src, dst, rotMat, src.size());
        std::string outputFilename = inputPath.stem().string() + "_rot_" + std::to_string(static_cast<int>(angle)) + inputPath.extension().string();
        cv::imwrite((fs::path(outputFolder) / outputFilename).string(), dst);
    }
}
