# Feature Matching for Object Detection

This project explores and compares two different approaches for object detection: a classic **feature matching pipeline** using SIFT and FLANN, and the **Viola-Jones** algorithm. 

## Objective

To detect predefined objects (sugar box, mustard bottle, power drill) in images using:
- **SIFT + FLANN-based feature matching**
- **Viola-Jones object detection algorithm**

The project evaluates detection accuracy and mean IoU across methods and includes several image preprocessing techniques to improve performance.

### Prerequisites
- C++ compiler (tested with g++)
- CMake
- OpenCV

### Build Instructions
./build.sh

### Run the Program
./run.sh

### Other note
See the report file.
