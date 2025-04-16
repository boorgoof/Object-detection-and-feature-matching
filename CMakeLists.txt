cmake_minimum_required(VERSION 3.10)
project(MIDDLEPROJECT_COMPUTERVISION)

find_package(OpenCV REQUIRED)

add_executable(main code/src/main.cpp code/src/Dataset.cpp)
target_link_libraries(main ${OpenCV_LIBS})

