#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <opencv2/opencv.hpp>
#include "Label.h"
#include <map>


namespace Utils{
    namespace Loader{
        cv::Mat load_image(const std::string& filepath);
        std::vector<Label> load_label_file(const std::string& filepath);
        const size_t load_folder_images(const std::string& folderpath, std::vector<cv::Mat>& output_images, std::vector<cv::Mat>& output_masks);
        const size_t load_folder_images(const std::string& folderpath, std::vector<cv::Mat>& output_images);
        std::vector<std::vector<Label>> load_folder_labels(const std::string& folderpath);
    }

    namespace Directory{
        std::vector<std::string> get_folder_filenames(const std::string& folderpath);
    }

    namespace String{
        const size_t split_string(const std::string& str, std::vector<std::string>& tokens, char delimiter);
        const std::string get_file_raw_basename(const std::string& filepath, char delimiter);
    }

    //https://stackoverflow.com/questions/54398336/stl-type-for-mapping-one-to-one-relations
    namespace Map{
        template <typename MapA2B, typename MapB2A = std::map<typename MapA2B::mapped_type, typename MapA2B::key_type>>
        MapB2A createInverseMap(const MapA2B& map){
            MapB2A inverseMap;
            for (const auto& pair : map) {
                inverseMap.emplace(pair.second, pair.first);
            }
            return inverseMap;
        }
    }
};

#endif