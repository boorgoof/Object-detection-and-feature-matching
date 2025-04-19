#include "../include/Dataset.h"
#include "../include/CustomErrors.h"
#include <iostream>

Dataset::Dataset(const Object_Type& type, const std::string& folderpath)
    : type{type}, folderpath{folderpath} {
    //if the folderpath is empty, the dataset is not loaded
    if(this->folderpath == "") return;

    this->load_test_items(folderpath);
    this->load_models(folderpath);
}

const size_t Dataset::load_test_items(const std::string& folderpath){
    this->test_items.clear();
    //loads the folder names in the given path (i.e. labels, models and test_images)
    std::vector<std::string> folders = Utils::Directory::get_folder_filenames(folderpath);
    
    //iterators to store the folder paths of labels and test_images
    std::vector<std::string>::iterator label_folder = folders.end();
    std::vector<std::string>::iterator test_images_folder = folders.end();

    for(std::vector<std::string>::iterator it=folders.begin(); it != folders.end(); ++it){
        //splits the folder path into directories tokens (removes the path, keeping only the last directory name)
        std::vector<std::string> tokens;
        const size_t n_t = Utils::String::split_string(*it, tokens, '/');   //splits folderpath into directories tokens

        //if the last directory is "labels" or "test_images", stores current folderpath (complete path) in label_folder or test_images_folder
        if(label_folder == folders.end()) label_folder = tokens[n_t-1].compare("labels")==0 ? it : folders.end();
        if(test_images_folder == folders.end()) test_images_folder = tokens[n_t-1].compare("test_images")==0 ? it : folders.end();
        
    }
    //if the last directory is not "labels" or "test_images", the folder is not valid
    if(label_folder == folders.end() || test_images_folder == folders.end()) throw CustomErrors::MissingDirectoryError(folderpath, "MISSING DIRECTORY, DATASET IS MALFORMED");

    //loads the filenames in the label and test_images folders
    std::vector<std::string> labels_filenames = Utils::Directory::get_folder_filenames(*label_folder);
    std::vector<std::string> test_images_filenames = Utils::Directory::get_folder_filenames(*test_images_folder);

    //iterates through label and images filenames
    std::vector<std::string>::iterator it_i = test_images_filenames.begin();
    for(std::vector<std::string>::iterator it_l = labels_filenames.begin(); it_l != labels_filenames.end(); ++it_l){
        //raw filenames (eliminate path, keep just the file name deleting the extension and the suffix (i.e. -color or -box))
        const std::string test_image_file_basename = Utils::Directory::get_file_basename(*it_i);
        const std::string test_image_file_name_raw = Utils::Directory::remove_file_suffix(test_image_file_basename,'-');

        const std::string label_file_basename = Utils::Directory::get_file_basename(*it_l);
        const std::string label_file_name_raw = Utils::Directory::remove_file_suffix(label_file_basename,'-');

        //if the 2 row filenames are the same then the pair label-image is correct
        if(test_image_file_name_raw.compare(label_file_name_raw) == 0){
            //load label and store image filename in vector of items
            this->test_items.push_back(std::pair<std::vector<Label>, std::string>(Utils::Loader::load_label_file((*it_l)), *it_i));
        }
        else throw CustomErrors::ImageLabelMismatch((*it_i),(*it_l), "IMAGE FILENAME AND LABEL FILENAME MISMATCH");
        ++it_i;
    }

    return this->test_items.size();
}

const size_t Dataset::load_models(const std::string& folderpath){
    this->models.clear();

    //loads the folder names in the given path (i.e. labels, models and test_images)
    std::vector<std::string> folders = Utils::Directory::get_folder_filenames(folderpath);
    
    //iterator to store the folder path of model folder
    std::vector<std::string>::iterator models_folder = folders.end();

    for(std::vector<std::string>::iterator it=folders.begin(); it != folders.end(); ++it){
        //splits the folder path into directories tokens (removes the path, keeping only the last directory name)
        std::vector<std::string> tokens;
        const size_t n_t = Utils::String::split_string(*it, tokens, '/');   //splits folderpath into directories tokens

        //if the last directory is "models", stores current folderpath (complete path) in models_folder
        if(models_folder == folders.end()) models_folder = tokens[n_t-1].compare("models")==0 ? it : folders.end();
    }
    //if the last directory is not "models", the folder is not valid
    if(models_folder == folders.end()) throw CustomErrors::MissingDirectoryError(folderpath, "MISSING DIRECTORY, DATASET IS MALFORMED");

    //splits model folder's files into two vectors: one contains colored images' filenames, the other contains masks' filenames
    std::vector<std::string> images_filenames;
    std::vector<std::string> masks_filenames;
    Utils::Directory::split_model_img_masks(*models_folder, images_filenames, masks_filenames);
    
    //if the number of images and masks is not the same, the folder is not valid
    if(images_filenames.size() != masks_filenames.size()) throw CustomErrors::ImageMaskMismatch("","", "IMAGES AND MASKS ARE NOT THE SAME NUMBER");

    //iterates through images and masks filenames
    std::vector<std::string>::iterator it_i = images_filenames.begin();
    for(std::vector<std::string>::iterator it_m = masks_filenames.begin(); it_m != masks_filenames.end(); ++it_m){
        //raw filenames (eliminate path, keep just the file name deleting the extension and the suffix (i.e. _color or _mask))
        std::string image_file_basename = Utils::Directory::get_file_basename(*it_i);
        std::string image_file_name_raw = Utils::Directory::remove_file_suffix(image_file_basename,'_');
        std::string mask_file_basename = Utils::Directory::get_file_basename(*it_m);
        std::string mask_file_name_raw = Utils::Directory::remove_file_suffix(mask_file_basename,'_');
        
        //if the 2 row filenames are the same then the pair mask-image is correct
        if(mask_file_name_raw.compare(image_file_name_raw) == 0){
            //load mask and store image filename in vector of models
            this->models.push_back(std::pair<std::string, std::string>(*it_i, *it_m));
        }
        else throw CustomErrors::ImageLabelMismatch((*it_i),(*it_m), "IMAGE FILENAME AND LABEL FILENAME MISMATCH");
        ++it_i;
    }

    return 0;
}

std::ostream& operator<<(std::ostream& os, const Dataset& d){
    os << d.get_folderpath() <<" type: " << d.get_type() << " #test items: " << d.get_items().size() << " #models " << d.get_models().size();
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

std::ostream& operator<<(std::ostream& os, const std::pair<std::string, std::string>& p){
    os << "image file path: \n\t" << p.first << "\nmask file path:\n\t" << p.second;

    return os;
}