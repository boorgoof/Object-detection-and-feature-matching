#ifndef IMAGEFILTER_H
#define IMAGEFILTER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <functional>
#include <utility>


class ImageFilter{
    private:
    std::vector<std::pair<std::string, std::function<cv::Mat(cv::Mat)>>> filter_pipeline;

    public:
    ImageFilter(){}

    cv::Mat apply_filters(const cv::Mat& src_img);

    template<typename FilterFunction, typename... Args>
    void add_filter(const std::string& filter_name, FilterFunction filter_function, Args&&... args);

    bool remove_filter(const std::string& filter_name);
};

namespace Filters{
    
    cv::Mat gaussian_blur(const cv::Mat& src_img, const cv::Size& kernel_size);
    cv::Mat median_blur(const cv::Mat& src_img, const cv::Size& kernel_size);
    cv::Mat average_blur(const cv::Mat& src_img, const cv::Size& kernel_size);

}

template<typename FilterFunction, typename... Args>
void ImageFilter::add_filter(const std::string& filter_name, FilterFunction filter_function, Args&&... args){
    auto packaged_function = [filter_function, args...](const cv::Mat& img) {
        return filter_function(img, args...);
    };
    this->filter_pipeline.push_back(std::make_pair(filter_name, packaged_function));
}

#endif