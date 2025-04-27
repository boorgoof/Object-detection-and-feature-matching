#include "../include/Label.h"
#include "../include/Utils.h"

std::map<std::string, Object_Type::Type> Object_Type::associative_map = { {"004_sugar_box",Object_Type::Type::SUGAR_BOX},
                                                            {"006_mustard_bottle", Object_Type::Type::MUSTARD_BOTTLE},
                                                            {"035_power_drill", Object_Type::Type::POWER_DRILL}};

std::map<Object_Type::Type, std::string> Object_Type::associative_map_reverse = Utils::Map::createInverseMap(Object_Type::associative_map);

const Object_Type::Type& Object_Type::string_to_enum(const std::string& type) {
    return associative_map[type];
}

const std::string& Object_Type::to_string() const {
    return associative_map_reverse[this->get_type()];
}

bool operator<(const Object_Type& lhs, const Object_Type& rhs){
    return lhs.get_type() < rhs.get_type();
}

std::ostream& operator<<(std::ostream& os, const Label& l){
    os << l.get_class_name() << " " << l.get_bounding_box();
    return os;
}

std::ostream& operator<<(std::ostream& os, const Object_Type& o){
    os << o.to_string();
    return os;
}