#include "../../../include/ObjectDetector/FeaturePipeline/ImageFilter.h"
#include "../../../include/CustomErrors.h"

cv::Mat ImageFilter::apply_filters(const cv::Mat& src_img){
    cv::Mat filtered_img = src_img.clone();
    cv::Mat filtered_image;
    for(auto& filter : this->filter_pipeline){
        //std::cout << "Applying filter: " << filter.first << std::endl;
        filtered_image = filter.second(filtered_img);
    }
    return filtered_image;
}

bool ImageFilter::remove_filter(const std::string& filter_name){
    this->filter_pipeline.erase(std::remove_if(this->filter_pipeline.begin(), this->filter_pipeline.end(),
        [&filter_name](const std::pair<std::string, std::function<cv::Mat(cv::Mat)>>& filter) {
            return filter.first == filter_name;
        }), this->filter_pipeline.end());

    return false;
}

cv::Mat Filters::gaussian_blur(const cv::Mat& src_img, const cv::Size& kernel_size){
    if (kernel_size.width <= 0 || kernel_size.height <= 0 || kernel_size.width % 2 == 0 || kernel_size.height % 2 == 0) {
        throw CustomErrors::InvalidArgumentError("kernel_size", "Gaussian kernel dimensions must be positive and odd.");
    }
    cv::Mat dst_img;
    cv::GaussianBlur(src_img, dst_img, kernel_size, 0);
    return dst_img;
}
cv::Mat Filters::median_blur(const cv::Mat& src_img, const cv::Size& kernel_size){
    if (kernel_size.width <= 0 || kernel_size.height <= 0 || kernel_size.width % 2 == 0 || kernel_size.height % 2 == 0) {
        throw CustomErrors::InvalidArgumentError("kernel_size", "Median kernel dimensions must be positive and odd.");
    }
    cv::Mat dst_img;
    cv::medianBlur(src_img, dst_img, kernel_size.width);
    return dst_img;
}
cv::Mat Filters::average_blur(const cv::Mat& src_img, const cv::Size& kernel_size){
    if (kernel_size.width <= 0 || kernel_size.height <= 0 || kernel_size.width % 2 == 0 || kernel_size.height % 2 == 0) {
        throw CustomErrors::InvalidArgumentError("kernel_size", "Average kernel dimensions must be positive and odd.");
    }
    cv::Mat dst_img;
    cv::blur(src_img, dst_img, kernel_size);
    return dst_img;
}

