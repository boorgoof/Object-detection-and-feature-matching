#include "Dataset.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string.h>
#include <map>
#include <iostream>

std::map<std::string, Object_Type::Type> Object_Type::associative_map = { {"004_sugar_box",Object_Type::Type::SUGAR_BOX},
                                                            {"006_mustard_bottle", Object_Type::Type::MUSTARD_BOTTLE},
                                                            {"035_power_drill", Object_Type::Type::POWER_DRILL}};

const Object_Type::Type& Object_Type::string_to_enum(const std::string& type){
    return associative_map[type];
}

size_t split_string(const std::string& str, std::vector<std::string>& tokens, char delimiter){
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

std::vector<std::string> get_folder_filenames(const std::string& folderpath){
    std::vector<std::string> filenames;
    for (const auto & entry : std::filesystem::directory_iterator(folderpath)) {
        filenames.push_back(entry.path());
    }

    if(filenames.size() == 0) throw std::runtime_error("NO FILES IN FOLDER: "+folderpath);

    return filenames;
}

cv::Mat LoaderUtils::load_image(const std::string& filepath) {
    cv::Mat img = cv::imread(filepath);
    if(img.empty()) throw std::runtime_error("COULD NOT OPEN FILE: "+filepath);
    return cv::imread(filepath);
}
std::vector<Label> LoaderUtils::load_label_file(const std::string& filepath){
    std::vector<Label> labels;

    std::string line = "";
    std::ifstream myfile (filepath);
    if (myfile.is_open()){
        while(getline(myfile, line)) {
            std::vector<std::string> tokens;
            size_t n = split_string(line, tokens, ' ');

            if(n!=5) throw std::runtime_error("LABEL FILE LINE DOES NOT MATCH LABEL FORMAT (class_name p1x p1y p2x p2y): "+line);
            cv::Point2i p1(std::stoi(tokens[1]), std::stoi(tokens[2]));
            cv::Point2i p2(std::stoi(tokens[3]), std::stoi(tokens[4]));
            labels.push_back(Label(Object_Type(tokens[0]), cv::Rect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y)));
        }
        myfile.close();
    }
    else throw std::runtime_error("COULD NOT OPEN FILE: "+filepath);

    return labels;
}
std::vector<cv::Mat> LoaderUtils::load_folder_images(const std::string& folderpath){
    std::vector<std::string> filenames = get_folder_filenames(folderpath);
    std::vector<cv::Mat> images;

    for(auto it = filenames.begin(); it != filenames.end(); ++it){
        images.push_back(load_image(*it));
    }

    return images;
}
std::vector<Label> LoaderUtils::load_folder_labels(const std::string& folderpath){
    std::vector<std::string> filenames = get_folder_filenames(folderpath);
    std::vector<Label> labels;

    for(auto it = filenames.begin(); it != filenames.end(); ++it){
        std::vector<Label> l = load_label_file(*it);
        labels.insert(labels.end(), l.begin(), l.end());
    }

    return labels;
}