# K-Nearest Neighbiour (kNN) Classifier

###### Problem Statement
Implement the k-Nearest  algorithm. We have faces.mat contains the Matlab variables traindata
(training data), trainlabels (training labels), testdata (test data), testlabels (test labels) and evaldata
(evaluation data, needed later). This is a facial attractiveness classification task: given a picture
of a face, you need to predict whether the average rating of the face is hot or not.

Pseudocode

```
a. Load the faces.mat file in python and extract the training and testing data.
b. For each value of k
c. For each training and testing instance
d. Find its k neighbours using cosine distance
e. Find the label corresponding to its neighbours based on voting
f. Calculate the error based on this predicted label.

```
![alt text](https://github.com/luas13/Machine_Learning/blob/master/KNN/kNN_error_rate.png)

One question is good to analyze is, does the value of k which minimizes the training error also minimize the test error.

## Answer

As per the graph above, the value of k which minimizes the training error is 1 while the value of k which minimizes the value of testing error is 20. Therefore we conclude that we should never choose ’k’ based on training error.
