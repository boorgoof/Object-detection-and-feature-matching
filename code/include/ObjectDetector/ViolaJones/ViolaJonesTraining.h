#ifndef ViolaJonesTraining_h
#define ViolaJonesTraining_h

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <random>
#include "Utils.h"


class ViolaJonesTraining {
public:
    int PositiveSamplesFiles(const std::string& , const std::string&, const std::string& );
    int NegativeSamplesFiles(const std::vector<std::string>& , const std::string& , const std::string& );
};

#endif