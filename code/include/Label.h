#ifndef LABEL_H
#define LABEL_H

#include <string>
#include <map>
#include <opencv2/opencv.hpp>

/**
 * @brief class to represent the type of an object (e.g. SUGAR_BOX, MUSTARD_BOTTLE, etc.).
 */
class Object_Type{
    public:
    /**
     * @brief enum to represent the different types of objects.
     */
    enum class Type {
        SUGAR_BOX = 4,
        MUSTARD_BOTTLE = 6,
        POWER_DRILL = 35
    };
    /**
     * @brief constructor to create an Object_Type object from a string (e.g. "004_sugar_box").
     */
    Object_Type(const std::string& type) : type{string_to_enum(type)} {};
    /**
     * @brief constructor to create an Object_Type object from enum Type (e.g. SUGAR_BOX).
     */
    Object_Type(const Type& type) : type{type} {};
    /**
     * @brief method to convert the enum Type to a string (e.g. "004_sugar_box").
     */
    const std::string& to_string() const;
    const Type& get_type() const {return this->type;}
    void set_type(const Type& new_type) {this->type = new_type;}
    /**
     * @brief sets the type of the object from a string (e.g. "004_sugar_box").
     * @param type the string representation of the type
     * @note this method uses the string_to_enum method to convert the string to enum Type.
     */
    void set_type(const std::string& type) {this->type = string_to_enum(type);}

    /**
     * @brief method to convert the string representation of the type to enum Type.
     */
    static const Type& string_to_enum(const std::string& type);
    /**
     * @brief mapping the string representation of the type to enum Type.
     */
    static std::map<std::string, Type> associative_map;
    /**
     * @brief mapping the enum Type to the string representation of the type.
     */
    static std::map<Type, std::string> associative_map_reverse;
    
    private:
    Type type;
};

bool operator<(const Object_Type& lhs, const Object_Type& rhs);

/**
 * @brief class to represent a label for an object in an image.
 */
class Label{
    private:
    /**
     * @brief the type of the object
     */
    Object_Type class_name;
    /**
     * @brief the bounding box of the object in the image
     */
    cv::Rect boundingbox;
    public:
    /**
     * @brief constructor to create a Label object from an Object_Type and a bounding box.
     * @param object_type the type of the object
     * @param boundingbox the bounding box of the object in the image
     */
    Label(const Object_Type& object_type, const cv::Rect& boundingbox) : class_name{object_type}, boundingbox{boundingbox} {}
    const Object_Type& get_class_name() const {return this->class_name;}
    const cv::Rect& get_bounding_box() const {return this->boundingbox;}
    void set_class_name(const Object_Type& new_name) {this->class_name = new_name;}
    void set_bounding_box(const cv::Rect& new_bb) {this->boundingbox = new_bb;}
};

std::ostream& operator<<(std::ostream& os, const Label& l);
std::ostream& operator<<(std::ostream& os, const Object_Type& o);

#endif