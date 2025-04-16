#ifndef DATASET_H
#define DATASET_H

#include "Label.h"
#include "Utils.h"
#include <map>


class Dataset{
    private:
    Object_Type type;   //object type of the dataset, assuming to create one dataset for every object type (sugar box, mustard, power drill)
    std::string folderpath;     //folder where the physical files are located (to check if another dataset is the copy of this one)
    std::vector<std::pair<std::vector<Label>, std::string>> items;   //vector of pairs of labels and image filepath (one image can have more labels since more than one object can appear)

    public:
    Dataset(const Object_Type& type, const std::string& folderpath = "");
    Dataset(const Dataset&) = delete;
    Dataset(Dataset&&) = delete;
    Dataset& operator=(const Dataset&) = delete;
    Dataset& operator=(Dataset&&) = delete;

    const size_t load_items(const std::string& folderpath);

    const Object_Type& get_type() const {return this->type;}
    const std::string& get_folderpath() const {return this->folderpath;}
    const std::vector<std::pair<std::vector<Label>, std::string>>& get_items() const {return this->items;}
};

std::ostream& operator<<(std::ostream& os, const Dataset& d);

std::ostream& operator<<(std::ostream& os, const std::pair<std::vector<Label>, std::string>& p);

std::ostream& operator<<(std::ostream& os, const std::vector<Label>& v);

#endif