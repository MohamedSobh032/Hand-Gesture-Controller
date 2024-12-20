# Hand Gesture Controller
This project contains 2 sections, Classical approach with Machine Learning, and Neural networks and Deep Learning Approach

## Algorithms Used in Classical Approach
- `K-Means Binary Clustering` - For hand segmentation.
- `Histogram of Gradients` - For feature sets.
- `SVM, KNN, RandomForest, Decision Tree, Naive Bayes` - For model generation.

## Pipeline Explanation in Classical Approach
- `images_generator.py` - Generates all the segmented images in a data directory to train the model.
- `dataset_generator.py` - Generates a feature dataset from the images generated.
- `model_generator.py` - Trains 4 modes (SVM, KNN, RandomForest, Decision Tree, Naive Bayes), find the best accuray and generates the model.
- `classifier.py` - The main application, segment the image and generates a HOG for it, then predicts using the model trained.

## References
- https://www.researchgate.net/publication/286480276_Hand_Gesture_Segmentation_Method_Based_on_YCbCr_Color_Space_and_K-Means_Clustering

## Needed Packages
To run the Hand Gesture Controller, you'll need the following Python packages:

- `opencv-python` - For image processing and computer vision tasks
- `os` - For creating paths that work on any OS
- `numpy` - For array handling and mathematical operations
- `pickle` - For data serialization in dataset creation and model generation
- `skimage` - For Histogram of Gradients function
- `sklearn` - For classifiers and model training
- `matplotlib` - For comparison and visualization of different classifiers accuray
- `pyautogui` - for controlling the mouse and keyboard

You can install these required packages using pip. Run the following command in your terminal:

```bash
pip install -r requirements.txt