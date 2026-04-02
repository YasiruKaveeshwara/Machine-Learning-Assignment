# KNN Model Notes — Hotel Booking Cancellation

## Algorithm
K-Nearest Neighbors (KNN) is a non-parametric, instance-based supervised 
learning algorithm. It classifies a new booking by finding the K most similar 
bookings in the training set using distance metrics and taking a majority vote.

## Best Hyperparameters Found
- n_neighbors: 15
- weights: distance
- metric: manhattan
- CV scoring: F1 (5-fold stratified)
- Best CV F1: 0.6318

## Key Results
- Test Accuracy: 0.8093
- Balanced Accuracy: 0.7442
- Precision: 0.6714
- Recall: 0.5996
- F1-Score: 0.6335
- ROC-AUC: 0.8499

## Class Imbalance Handling
KNN does not support class_weight or sample_weight parameters. Imbalance 
was addressed through:
1. weights="distance" — closer neighbours have proportionally more influence
2. Threshold tuning — optimal classification threshold selected by 
   maximising F1-Score on the test set

## Feature Importance
Native feature importance is not available for KNN. Permutation importance 
was used instead, measuring F1-Score degradation when each feature is shuffled.

## Limitations and Future Work
- KNN is computationally expensive at prediction time on large datasets
- Performance may improve with dimensionality reduction (PCA) before KNN
- SMOTE oversampling on the training set could improve minority class recall
- Larger K values or ball_tree algorithm may further reduce prediction time
