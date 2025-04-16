#include "../include/Dataset.h"
#include "../include/CustomErrors.h"
#include <iostream>

Dataset::Dataset(const Object_Type& type, const std::string& folderpath)
    : type{type}, folderpath{folderpath} {
    if(this->folderpath == "") return;

    this->load_items(folderpath);
}

const size_t Dataset::load_items(const std::string& folderpath){
    this->items.clear();

    std::vector<std::string> folders = Utils::Directory::get_folder_filenames(folderpath);
    
    std::vector<std::string>::iterator label_folder = folders.end();
    std::vector<std::string>::iterator test_images_folder = folders.end();

    for(std::vector<std::string>::iterator it=folders.begin(); it != folders.end(); ++it){
        std::vector<std::string> tokens;
        const size_t n_t = Utils::String::split_string(*it, tokens, '/');   //splits folderpath into directories tokens

        if(label_folder == folders.end()) label_folder = tokens[n_t-1].compare("labels")==0 ? it : folders.end();  //if last directory == "labels" stores current folderpath (complete) in label_folder
        if(test_images_folder == folders.end()) test_images_folder = tokens[n_t-1].compare("test_images")==0 ? it : folders.end();  //same but change the directory name

    }
    
    if(label_folder == folders.end() || test_images_folder == folders.end()) throw CustomErrors::MissingDirectoryError(folderpath, "MISSING DIRECTORY, DATASET IS MALFORMED");

    //vectors of subdirectories
    std::vector<std::string> labels_filenames = Utils::Directory::get_folder_filenames(*label_folder);
    std::vector<std::string> test_images_filenames = Utils::Directory::get_folder_filenames(*test_images_folder);

    //iterates through label and images filenames
    std::vector<std::string>::iterator it_i = test_images_filenames.begin();
    for(std::vector<std::string>::iterator it_l = labels_filenames.begin(); it_l != labels_filenames.end(); ++it_l){
        //raw filenames (eliminate path, keep just the file name deleting the extension and the suffix (i.e. -color or -box))
        const std::string test_image_file_basename = Utils::String::get_file_raw_basename(*it_i,'-');
        const std::string label_file_basename = Utils::String::get_file_raw_basename(*it_l,'-');
        //if the 2 row filenames are the same -> the pair of label-image is correct
        if(test_image_file_basename.compare(label_file_basename) == 0){
            //filenames match, load label and store image filename in vector of items
            this->items.push_back(std::pair<std::vector<Label>, std::string>(Utils::Loader::load_label_file((*it_l)), *it_i));
        }
        else throw CustomErrors::ImageLabelMismatch((*it_i),(*it_l), "IMAGE FILENAME AND LABEL FILENAME MISMATCH");
        ++it_i;
    }

    return this->items.size();
}

std::ostream& operator<<(std::ostream& os, const Dataset& d){
    os << d.get_folderpath() <<" type: " << d.get_type() << " #items: " << d.get_items().size();
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::pair<std::vector<Label>, std::string>& p){
    os << "image file path: \n\t" << p.second << "\nbounding boxes:\n" << p.first;

    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Label>& v){
    for(auto it = v.begin(); it != v.end(); ++it){
        os << "\t" << *it << std::endl;
    }

    return os;
}