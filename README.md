# MiddleProject_ComputerVision

Task:
- image loader module.
- algorithm for detecting features of the images
    - SIFT (Best one ?)
    - Surf
    - Fast
    - Brief
    - Orb
    - Gloh
- algorithm that given features find them on the image.
    - Viola & Jones algorithm (Haar cascade)
    - Brute force
    - Flann based matcher
- Bounding box
    - https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches
    - Viola & Jones automatic bounding box

# Report

DetectObject is an abstract class that generally defines the ability of a class to identify an object from an image, returning a Label. A Label DetectObject is an abstract class that generally defines the ability of a class to identify an object from an image, returning a Label. A Label specifies the type of the returned object and the corresponding bounding box in the reference image.

So two classes have been defined that, by extending the Object detector, try to solve the problem of identifying an object with different approaches. 
Specifically, the implemented classes are FeaturePipeline and ViolaAndJones.

### FeaturePipeline:

FeaturePipeline overrides the virtual function DetectObjects by implementing a system based on feature matching. 
In particular, the main steps performed by the FeaturePipeline class are the following:

- First, once the object of interest has been chosen, extracting the features of each available model is necessary. This is an operation performed in advance, that is, before the invocation of the detect_object function, to allow the pipeline to know the characteristics of the object to be identified.

- We then proceed by extracting the features of the image on which we want to identify the object.

- Now the matches between the features of the models and the target image are extracted, to select the best model, that is, the one that produces the highest number of “good” matches evaluated through Lowe’s test.

- Finally, we proceed using the matches of the best model to identify the bounding box of the object in the image.

The returned label will be evaluated in terms of accuracy and used to trace the identifying rectangle of the object in the image. The result will then be recorded in the output folder.

