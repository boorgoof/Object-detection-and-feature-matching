#include <string>
#include <opencv2/opencv.hpp>

class Object_Type{
    public:
    enum class Type {
        SUGAR_BOX = 4,
        MUSTARD_BOTTLE = 6,
        POWER_DRILL = 35
    };
    Object_Type(const std::string& type) : type{string_to_enum(type)} {};
    Object_Type(const Type& type) : type{type} {};
    const std::string& to_string();
    const Type& get_type() const {return this->type;}
    void set_type(const Type& new_type) {this->type = new_type;}
    void set_type(const std::string& type) {this->type = string_to_enum(type);}

    static const Type& string_to_enum(const std::string& type);
    static std::map<std::string, Type> associative_map;

    private:
    Type type;
};

class Label{
    private:
    Object_Type class_name;
    cv::Rect bb;
    public:
    Label(const Object_Type& object_type, const cv::Rect& bb) : class_name{object_type}, bb{bb} {}
    const Object_Type& get_class_name() const {return this->class_name;}
    const cv::Rect& get_bounding_box() const {return this->bb;}
    void set_class_name(const Object_Type& new_name) {this->class_name = new_name;}
    void set_bounding_box(const cv::Rect& new_bb) {this->bb = new_bb;}
};

size_t split_string(const std::string& str, std::vector<std::string>& tokens, char delimiter);
std::vector<std::string> get_folder_filenames(const std::string& folderpath);


namespace LoaderUtils{

    cv::Mat load_image(const std::string& filepath);
    std::vector<Label> load_label_file(const std::string& filepath);
    std::vector<cv::Mat> load_folder_images(const std::string& folderpath);
    std::vector<Label> load_folder_labels(const std::string& folderpath);

};