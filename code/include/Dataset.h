#ifndef DATASET_H
#define DATASET_H

#include "Label.h"
#include "Utils.h"
#include <map>


class Dataset{
    private:
    Object_Type type;   //object type of the dataset, assuming to create one dataset for every object type (sugar box, mustard, power drill)
    std::string folderpath;     //folder where the physical files are located (to check if another dataset is the copy of this one)
    std::vector<std::pair<std::vector<Label>, std::string>> test_items;   //vector of pairs of labels and image filepath (one image can have more labels since more than one object can appear)
    std::vector<std::pair<std::string, std::string>> models; //pair of IMAGE COLOR - MASK

    const size_t load_models_filenames(const std::string& folderpath, std::vector<std::string>& images_filenames, std::vector<std::string>& masks_filenames);

    public:
    Dataset(const Object_Type& type, const std::string& folderpath = "");

    const size_t load_test_items(const std::string& folderpath);
    const size_t load_models(const std::string& folderpath);

    const Object_Type& get_type() const {return this->type;}
    const std::string& get_folderpath() const {return this->folderpath;}
    const std::vector<std::pair<std::vector<Label>, std::string>>& get_items() const {return this->test_items;}
    const std::vector<std::pair<std::string, std::string>>& get_models() const {return this->models;}
};

std::ostream& operator<<(std::ostream& os, const Dataset& d);

std::ostream& operator<<(std::ostream& os, const std::pair<std::vector<Label>, std::string>& p);

std::ostream& operator<<(std::ostream& os, const std::vector<Label>& v);

std::ostream& operator<<(std::ostream& os, const std::pair<std::string, std::string>& p);

#endif