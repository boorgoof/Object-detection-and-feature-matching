#include "../include/Utils.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string.h>
#include <iostream>
#include "../include/CustomErrors.h"


const size_t Utils::String::split_string(const std::string& str, std::vector<std::string>& tokens, char delimiter){
    size_t pos = str.find( delimiter );
    size_t initialPos = 0;
    tokens.clear();

    while( pos != std::string::npos ) {
        tokens.push_back(str.substr( initialPos, pos - initialPos ) );
        initialPos = pos + 1;

        pos = str.find( delimiter, initialPos );
    }

    tokens.push_back( str.substr( initialPos, std::min( pos, str.size() ) - initialPos + 1 ) );

    return tokens.size();
}

const std::string Utils::Directory::get_file_basename(const std::string& filepath){
    //gets the file name from the full path
    std::string filename = filepath.substr(filepath.find_last_of("/")+1);
    //removes the file extension
    return filename.substr(0, filename.find_first_of('.'));
}

const std::string Utils::Directory::remove_file_suffix(const std::string& file_basename, char delimiter){
    return file_basename.substr(0, file_basename.find_last_of(delimiter));
}
std::vector<std::string> Utils::Directory::get_folder_filenames(const std::string& folderpath){
    std::vector<std::string> filenames;
    //gets all the files names in the specified folder (also subfolders)
    for (const auto & entry : std::filesystem::directory_iterator(folderpath)) {
        filenames.push_back(entry.path());
    }

    if(filenames.size() == 0) throw CustomErrors::EmptyFolderError(folderpath, "NO FILES IN FOLDER");
    //sorts the filenames alphabetically ascending
    std::sort(filenames.begin(), filenames.end());

    return filenames;
}

const size_t Utils::Directory::split_model_img_masks(const std::string& folderpath, std::vector<std::string>& images_filenames, std::vector<std::string>& masks_filenames){
    images_filenames.clear();
    masks_filenames.clear();
    //gets all the files names in the specified folder (assuming they are all files)
    std::vector<std::string> filenames = Utils::Directory::get_folder_filenames(folderpath);

    for(auto it=filenames.begin(); it != filenames.end(); ++it){
        std::string file_basename = Utils::Directory::get_file_basename(*it);

        //matches the output array with the file suffix
        if(file_basename.find("mask") != std::string::npos) masks_filenames.push_back(*it);
        else if(file_basename.find("color") != std::string::npos) images_filenames.push_back(*it);
        else throw CustomErrors::FileNameError(*it, "FILE HAS TO END WITH mask OR color TO BE A SUITABLE IMAGE");
    }

    std::sort(images_filenames.begin(), images_filenames.end());
    std::sort(masks_filenames.begin(), masks_filenames.end());

    return images_filenames.size();
}

cv::Mat Utils::Loader::load_image(const std::string& filepath) {
    cv::Mat img = cv::imread(filepath);
    if(img.empty()) throw CustomErrors::ImageLoadError(filepath,"COULD NOT OPEN FILE");

    std::string base_filename = Utils::Directory::get_file_basename(filepath);

    //checks if the file name contains "mask" or "color" to determine if it is a mask or a colored image and loads it accordingly
    if(base_filename.find("mask") != std::string::npos){
        img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    }
    else if(base_filename.find("color") != std::string::npos){
        img = cv::imread(filepath, cv::IMREAD_COLOR);
    }
    else throw CustomErrors::FileNameError(base_filename, "FILE HAS TO END WITH mask OR color TO BE A SUITABLE IMAGE");

    return img;
}
std::vector<Label> Utils::Loader::load_label_file(const std::string& filepath){
    std::vector<Label> labels;

    std::string line = "";
    std::ifstream myfile (filepath);
    if (myfile.is_open()){
        while(getline(myfile, line)) {
            std::vector<std::string> tokens;
            size_t n = Utils::String::split_string(line, tokens, ' ');

            if(n!=5) throw CustomErrors::LabelFormatError(line,"LABEL FILE LINE DOES NOT MATCH LABEL FORMAT (class_name p1x p1y p2x p2y)");

            //tokens 1 to 4 are the coordinates of the bounding box, token 0 is the object type name
            cv::Point2i p1(std::stoi(tokens[1]), std::stoi(tokens[2]));
            cv::Point2i p2(std::stoi(tokens[3]), std::stoi(tokens[4]));
            labels.push_back(Label(Object_Type(tokens[0]), cv::Rect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y)));
        }
        myfile.close();
    }
    else throw CustomErrors::ImageLoadError(filepath,"COULD NOT OPEN FILE");

    return labels;
}
const size_t Utils::Loader::load_folder_images(const std::string& folderpath, std::vector<cv::Mat>& output_images, std::vector<cv::Mat>& output_masks){
    output_images.clear();
    output_masks.clear();

    std::vector<std::string> filenames = Utils::Directory::get_folder_filenames(folderpath);

    for(auto it = filenames.begin(); it != filenames.end(); ++it){
        
        cv::Mat img = load_image(*it);
        //if loaded image has 1 channel, it is a mask, so it is pushed to the masks vector
        if(img.channels()==1){
            output_masks.push_back(img);
        }
        //else it is a colored image, so it is pushed to the images vector
        else if(img.channels()==3){
            output_images.push_back(img);
        }
        //if the image has more than 3 channels, it is not a valid image, so it is ignored
        else throw CustomErrors::ImageLoadError(*it,"IMAGE IS NEITHER A MASK NOR A COLORED IMAGE (TOO MANY CHANNELS)");
    }

    return output_images.size();
}
const size_t Utils::Loader::load_folder_images(const std::string& folderpath, std::vector<cv::Mat>& output_images){
    output_images.clear();

    std::vector<std::string> filenames = Utils::Directory::get_folder_filenames(folderpath);

    for(auto it = filenames.begin(); it != filenames.end(); ++it){
        output_images.push_back(load_image(*it));
    }

    return output_images.size();
}
std::vector<std::vector<Label>> Utils::Loader::load_folder_labels(const std::string& folderpath){
    std::vector<std::string> filenames = Utils::Directory::get_folder_filenames(folderpath);
    std::vector<std::vector<Label>> labels;

    for(auto it = filenames.begin(); it != filenames.end(); ++it){
        std::vector<Label> l = load_label_file(*it);
        labels.push_back(l);
    }

    return labels;
}