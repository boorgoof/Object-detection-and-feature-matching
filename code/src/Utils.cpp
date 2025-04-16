#include "../include/Utils.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string.h>
#include <iostream>
#include "../include/CustomErrors.h"

//https://stackoverflow.com/questions/5888022/split-string-by-single-spaces
const size_t Utils::String::split_string(const std::string& str, std::vector<std::string>& tokens, char delimiter){
    size_t pos = str.find( delimiter );
    size_t initialPos = 0;
    tokens.clear();

    while( pos != std::string::npos ) {
        tokens.push_back(str.substr( initialPos, pos - initialPos ) );
        initialPos = pos + 1;

        pos = str.find( delimiter, initialPos );
    }

    // Add the last one
    tokens.push_back( str.substr( initialPos, std::min( pos, str.size() ) - initialPos + 1 ) );

    return tokens.size();
}

std::vector<std::string> Utils::Directory::get_folder_filenames(const std::string& folderpath){
    std::vector<std::string> filenames;
    for (const auto & entry : std::filesystem::directory_iterator(folderpath)) {
        filenames.push_back(entry.path());
    }

    if(filenames.size() == 0) throw CustomErrors::EmptyFolderError(folderpath, "NO FILES IN FOLDER");

    return filenames;
}

cv::Mat Utils::Loader::load_image(const std::string& filepath) {
    cv::Mat img = cv::imread(filepath);
    if(img.empty()) throw CustomErrors::ImageLoadError(filepath,"COULD NOT OPEN FILE");

    std::string filename = filepath.substr(filepath.find_last_of("/")+1);
    std::string base_filename = filename.substr(0, filename.find_first_of('.'));

    if(base_filename.find("mask") != std::string::npos){
        img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    }
    else if(base_filename.find("color") != std::string::npos){
        img = cv::imread(filepath, cv::IMREAD_COLOR);
    }
    else throw CustomErrors::FileNameError(filename, "FILE HAS TO END WITH mask OR color TO BE A SUITABLE IMAGE");

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
            cv::Point2i p1(std::stoi(tokens[1]), std::stoi(tokens[2]));
            cv::Point2i p2(std::stoi(tokens[3]), std::stoi(tokens[4]));
            labels.push_back(Label(Object_Type(tokens[0]), cv::Rect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y)));
        }
        myfile.close();
    }
    else throw CustomErrors::ImageLoadError(filepath,"COULD NOT OPEN FILE");

    return labels;
}
//loads image folder separating colored images from masks
const size_t Utils::Loader::load_folder_images(const std::string& folderpath, std::vector<cv::Mat>& output_images, std::vector<cv::Mat>& output_masks){
    output_images.clear();
    output_masks.clear();

    std::vector<std::string> filenames = Utils::Directory::get_folder_filenames(folderpath);

    for(auto it = filenames.begin(); it != filenames.end(); ++it){
        
        cv::Mat img = load_image(*it);
        if(img.channels()==1){
            output_masks.push_back(img);
        }
        else output_images.push_back(img);
    }

    return output_images.size();
}
//loads image folder
const size_t Utils::Loader::load_folder_images(const std::string& folderpath, std::vector<cv::Mat>& output_images){
    output_images.clear();

    std::vector<std::string> filenames = Utils::Directory::get_folder_filenames(folderpath);

    for(auto it = filenames.begin(); it != filenames.end(); ++it){
        output_images.push_back(load_image(*it));
    }

    return output_images.size();
}
//loads labels in a folder, each element can contain more than one label
std::vector<std::vector<Label>> Utils::Loader::load_folder_labels(const std::string& folderpath){
    std::vector<std::string> filenames = Utils::Directory::get_folder_filenames(folderpath);
    std::vector<std::vector<Label>> labels;

    for(auto it = filenames.begin(); it != filenames.end(); ++it){
        std::vector<Label> l = load_label_file(*it);
        labels.push_back(l);
    }

    return labels;
}