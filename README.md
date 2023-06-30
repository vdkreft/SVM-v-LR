# SVM-v-LR

# Dependencies and Requirements
Model performance depends on the connectivity matrix used for classification algorithm construction. 1000x1000 connectivity matrices were used for our analysis. 

Download NeuroConn by @Victoris93 for the best results and data preparation for connectivity matrix use.

# Overview of Package
Data Preparation:
- Defining future functions

Support Vector Machine:
- No Hyperparameters 
- Hyperparameter gridsearch and use
- Permutation testing

Logistic Regression:
- Running simple LR
- Comparison to Support Vector Machine through ROC graph

Threshold and parcel count testing:
- 10% v. 5% thresholding
    - ANOVA
    - numpy
- Parcel count testing
- Confusion Matrix

# Note
All parts of the package will ask you to redefine pathways. Although this may be repetitive, it is designed for you to be able to only run certain portions rather than all lines of code. It is for convenience only. 

# Preperation before Running
You must have the following:
- Connectivity matrices in a folder named "Features_{n_parcels}" (ex. Features_1000)
- Connectivity matrices using the following convention: {feature_directory}z-conn-matrix-{participant}-rest-schaefer{n_parcels}.npy
- CSV file with participant_ids that have an associated connectivity matrix (this is used to makes sure your TSV files don't have unassociated participant IDs)
- TSV file with participant_id and diagnosis (other columns will be ignored)