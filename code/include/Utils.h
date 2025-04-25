#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <opencv2/opencv.hpp>
#include "Label.h"
#include <map>

/**
 * @brief utility functions.
 */
namespace Utils{
    /**
     * @brief functions to load images and labels.
     */
    namespace Loader{
        /**
         * @brief loads an image from a file, distringuishes between colored images and mask images by the file suffix (e.g. "-color" or "-mask").
         * @param filepath the path to the image file
         * @return the loaded image
         */
        cv::Mat load_image(const std::string& filepath);
        /**
         * @brief loads labels from a .txt file, where each line contains the label of an object in the format "object_type x y width height".
         * @param filepath the path to the label file
         * @return a vector of Label objects
         */
        std::vector<Label> load_label_file(const std::string& filepath);
        /**
         * @brief loads images and masks from a folder, separating colored images from masks, and sorting the 2 output vectors alphabetically ascending.
         * @param folderpath the path to the folder containing the images and masks
         * @param output_images the vector to store the loaded colored images
         * @param output_masks the vector to store the loaded masks
         * @return the number of loaded images (equal to the size of output_images)
         */
        const size_t load_folder_images(const std::string& folderpath, std::vector<cv::Mat>& output_images, std::vector<cv::Mat>& output_masks);
        /**
         * @brief loads images from a folder, doesn't separate colored images from masks.
         * @param folderpath the path to the folder containing the images
         * @param output_images the vector to store the loaded images
         * @return the number of loaded images
         */
        const size_t load_folder_images(const std::string& folderpath, std::vector<cv::Mat>& output_images);
        /**
         * @brief loads labels from a folder, each file can contain more than one label, so the output is a vector of vectors of Label objects.
         * @param folderpath the path to the folder containing the label files
         * @return a vector of vectors of Label objects
         */
        std::vector<std::vector<Label>> load_folder_labels(const std::string& folderpath);
    }

    /**
     * @brief functions to handle directories and file paths.
     */
    namespace Directory{
        /**
         * @brief gets the filenames of all files and folders in a folder.
         * @param folderpath the path to the folder
         * @return a vector of strings containing the filenames
         */
        std::vector<std::string> get_folder_filenames(const std::string& folderpath);
        /**
         * @brief removes the file path from a full file path, leaving only the file name and removing the extension.
         * @param filepath the full file path
         * @return the file name
         */
        const std::string get_file_basename(const std::string& filepath);
        /**
         * @brief removes the file suffix from a file name.
         * @param filepath the file name
         * @param delimiter the character used to separate the file name from the suffix
         * @return the file name without the suffix
         */
        const std::string remove_file_suffix(const std::string& filepath, char delimiter);
        /**
         * @brief given a folder path, it loads all the file names of images and masks in the folder, separating them into 2 vectors.
         * @param folderpath the path to the folder
         * @param images_filenames the vector to store the loaded images' file names
         * @param masks_filenames the vector to store the loaded masks' file names
         * @return the number of loaded images (equal to the size of images_filenames)
         */
        const size_t split_model_img_masks(const std::string& folderpath, std::vector<std::string>& images_filenames, std::vector<std::string>& masks_filenames);
    }

    /**
     * @brief functions to handle strings.
     */
    namespace String{
        /**
         * @brief splits a string into tokens using a delimiter.
         * @param str the string to be split
         * @param tokens the vector to store the tokens
         * @param delimiter the character used to split the string
         * @return the number of tokens
         * 
         * @note function gently retrieved from //https://stackoverflow.com/questions/5888022/split-string-by-single-spaces
         */
        const size_t split_string(const std::string& str, std::vector<std::string>& tokens, char delimiter);
    }

    /**
     * @brief functions to handle maps.
     */
    namespace Map{
        /**
         * @brief function to create an inverse map from a given map.
         * @tparam MapA2B the type of the map to be inverted
         * @tparam MapB2A the type of the inverted map
         * @param map the map to be inverted
         * @return the inverted map
         * 
         * @note function gently retrieved from //https://stackoverflow.com/questions/54398336/stl-type-for-mapping-one-to-one-relations
         */
        template <typename MapA2B, typename MapB2A = std::map<typename MapA2B::mapped_type, typename MapA2B::key_type>>
        MapB2A createInverseMap(const MapA2B& map){
            MapB2A inverseMap;
            for (const auto& pair : map) {
                inverseMap.emplace(pair.second, pair.first);
            }
            return inverseMap;
        }
    }

    /**
     * @brief Detection accuracy functions.
     */
    namespace DetectionAccuracy{
        double calculateIoU(const Label& predictedLabel, const Label& realLabel);
        double calculateMeanIoU(const Object_Type obj, std::map<std::string, std::vector<Label>>& realItems, const std::map<std::string, std::vector<Label>>& predictedItems);
        double calculateDatasetAccuracy(const Object_Type obj, std::map<std::string, std::vector<Label>>& realItems, const std::map<std::string, std::vector<Label>>& predictedItems , double threshold = 0.5);
        void printLabelsImg(const std::map<std::string, std::vector<Label>>& predictedItems, const std::map<std::string, std::vector<Label>>& realItems);
    };




};

#endif