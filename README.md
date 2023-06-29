# SVM-v-LR

# Dependencies and Requirements
Model performance depends on the connectivity matrix used for classification algorithm construction. 1000x1000 connectivity matrices were used for our analysis. 

Download NeuroConn by @Victoris93 for the best results and data preparation for connectivity matrix use.

# Overview of Package
Data Preparation:
- Defining future functions
- Locating pathways for necessary data

Support Vector Machine:
- No Hyperparameters 
- Hyperparameter gridsearch and use
- Permutation testing

Logistic Regression:
- Running simple LR
- Comparison to Support Vector Machine through ROC graph

Threshold and parcel count testing:
- 10% v. 5% thresholding
- Parcel count testing
- Confusion Matrix

# Preperation before Running
You must have the following:
- Connectivity matrices in a folder named "Features_{n_parcels}" (ex. Features_1000)
- Connectivity matrices using the following convention: {feature_directory}z-conn-matrix-{participant}-rest-schaefer{n_parcels}.npy
- Folder with names of subjects as individual file names (ex. sub-0001)
- CSV file with participant_id and diagnosis (other columns will be ignored)
