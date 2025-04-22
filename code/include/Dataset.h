#ifndef DATASET_H
#define DATASET_H

#include "Label.h"
#include "Utils.h"
#include <map>


/**
 * @brief Dataset class to manage a single object's test images, labels and models.
 */
class Dataset{
    private:
    /**
     * @brief Object_Type enum to define the type of object in the dataset.
     */
    Object_Type type;
    /**
     * @brief folderpath string to define the folder where the dataset is located.
     * @details used to check if another dataset is a copy of this one.
     * @note this is not used in the current implementation.
     */ 
    std::string folderpath;
    /**
     * @brief test_items vector of pairs of labels and image filepath.
     * @details one image can have more labels since more than one object can appear, also one image can contain objects (and labels) of other objects types.
     */
    std::vector<std::pair<std::vector<Label>, std::string>> test_items; //MODIFY, BUILD A MAP OF string(filepath test image) - label
    /**
     * @brief models vector of pairs of image and mask filepath (image first, mask second).
     * @details the image is the colored version of the object, the mask filters out the background of the image.
     */
    std::vector<std::pair<std::string, std::string>> models;
    /**
     * @brief load model filenames (divided into images and masks filenames) from the corresponding model folder (assuming that images and masks files are mixed in the folder).
     * @param folderpath string to define the model folder path.
     * @param images_filenames output vector of strings storing colored images file paths.
     * @param masks_filenames output vector of strings storing masks file paths.
     * @return size_t number of models loaded.
     * @details if the number of images and masks is not the same, the function will throw an error.
     */
    const size_t load_models_filenames(const std::string& folderpath, std::vector<std::string>& images_filenames, std::vector<std::string>& masks_filenames);

    public:
    /**
     * @brief Dataset constructor, loads test items' and models' filenames from the corresponding folders. 
     * @param type Object_Type enum to define the type of object in the dataset.
     * @param folderpath string to define the folder where the dataset is located.
     */
    Dataset(const Object_Type& type, const std::string& folderpath = "");

    /**
     * @brief method to load test items' filenames from the corresponding folder into class member test_items.
     * @param folderpath string to define the test items folder path.
     * @return size_t number of test items loaded.
     */
    const size_t load_test_items(const std::string& folderpath);
    /**
     * @brief method to load models' filenames from the corresponding folder into class member models.
     * @param folderpath string to define the models folder path.
     * @return size_t number of models loaded.
     * @details the function will throw an error if the number of images and masks is not the same.
     */
    const size_t load_models(const std::string& folderpath);

    /**
     * @brief return the object type of the dataset.
     * @return Object_Type enum to define the type of object in the dataset.
     */
    const Object_Type& get_type() const {return this->type;}
    /**
     * @brief return the folder path of the dataset.
     * @return string to define the folder where the dataset is located.
     */
    const std::string& get_folderpath() const {return this->folderpath;}
    /**
     * @brief return the test items of the dataset.
     * @return vector of pairs of labels and image filepath.
     */
    const std::vector<std::pair<std::vector<Label>, std::string>>& get_items() const {return this->test_items;}

    const std::map<std::string, std::vector<Label>> get_items_map() const; // TODO
    /**
     * @brief return the models of the dataset.
     * @return vector of pairs of image and mask filepath.
     */
    const std::vector<std::pair<std::string, std::string>>& get_models() const {return this->models;}
};

std::ostream& operator<<(std::ostream& os, const Dataset& d);

std::ostream& operator<<(std::ostream& os, const std::pair<std::vector<Label>, std::string>& p);

std::ostream& operator<<(std::ostream& os, const std::vector<Label>& v);

std::ostream& operator<<(std::ostream& os, const std::pair<std::string, std::string>& p);

namespace Utils::Loader{
    std::map<Object_Type, Dataset> load_datasets(const std::string& dataset_path);
}

#endif