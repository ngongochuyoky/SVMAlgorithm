# SVM Algorithm
## Introduction
Research on the application of computer vision techniques to the problem of classification of animal images, through photographs using self-built SVM (Support Vector Machine) classification method and SVM classification method using libraries sklearn. Thereby evaluating the optimization of the two models.
- The training data is an image set consisting of 3000 images with each class of 1000 images of cats, dogs and sheep respectively.
- Test data consists of 600 images, with each layer is 200 images.

Algorithm:

- SVM(Multi-Class Classification) : one-vs-one classification method
- Mini-batch gradient descent is a variation of the gradient descent algorithm
## File SVM.ipynb: 
- Split training data and test data
- Feature extraction
- Model training
- Predict
- Evaluate a self-built SVM model with a library-using SVM model
## Gains:
 - The program accurately predicts animals through images, with higher prediction accuracy than using the Sklearn library
 
## File app.py: 
 - Build a UI for Model using Streamlit
 
![UI](https://user-images.githubusercontent.com/54812014/220859105-853d8e07-64ff-4113-85f2-88100d5c8880.PNG)
