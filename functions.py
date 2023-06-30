import numpy as np

import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score

import nilearn
from nilearn.connectome import sym_matrix_to_vec

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#This function concates all datasets designated by dataset_paths as a single pandas dataframe.
def format_concate(dataset_paths):
    datasets = []
    for path in dataset_paths:
        dataset = pd.read_csv(path, sep='\t')
        datasets.append(dataset)

    for i in range(len(datasets)):
        dataset = datasets[i]

        if 'diag' in dataset.columns:
            dataset.rename(columns={'diag': 'diagnosis'}, inplace=True)
        elif 'dx' in dataset.columns:
            dataset.rename(columns={'dx': 'diagnosis'}, inplace=True)

        dataset = dataset[['participant_id', 'diagnosis']]

        dataset = dataset.replace(to_replace=["CONTROL", "No_Known_Disorder", 0], value="0")
        dataset = dataset.replace(to_replace=["SCHZ", "Schizophrenia_Strict", 4], value="1")
        dataset['participant_id'] = dataset['participant_id'].apply(lambda x: 'sub-' + x if not x.startswith('sub-') else x)

        dataset = dataset[(dataset['diagnosis'] == '0') | (dataset['diagnosis'] == '1')]

        datasets[i] = dataset

    y = pd.concat(datasets, ignore_index=True)

    return y

#This function filters out all participants that do not have a connectivity matrix
def filter_data(csv_file_path, y):
    csv_df = pd.read_csv(csv_file_path)

    csv_participant_ids = csv_df['participant_id'].tolist()

    matching_ids = y[y['participant_id'].isin(csv_participant_ids)]

    return(matching_ids)

#This function creates variables x (features), y (diagnosis), and the diagnosis key which is used later. 
def create_feat(feature_directory, n_parcels, matching_ids):
    all_features = []
    for participant in matching_ids['participant_id']:
        correlation_matrix = np.load(f"{feature_directory}/z-conn-matrix-{participant}-rest-schaefer{n_parcels}.npy")
        if len(correlation_matrix.shape) == 3:
            correlation_matrix = correlation_matrix[0, :, :]
        vec_correlation_matrix = nilearn.connectome.sym_matrix_to_vec(correlation_matrix, discard_diagonal=True)
        all_features.append(vec_correlation_matrix)

    np.savez_compressed(f'correlation_matrix{n_parcels}.npz',a = all_features)

    x = np.load(f'correlation_matrix{n_parcels}.npz')['a']
    y = matching_ids['diagnosis']
    y = y.to_numpy()
    diagnosis = pd.DataFrame(y)

    y = y.astype('int')

    return(x, y, diagnosis)

#Takes x, y, and diagnosis key and creates training and testing sets for machine learning.
def split(x, y, diagnosis):
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25, stratify= diagnosis, random_state=99)

    y_test = y_test.astype('int')
    y_train = y_train.astype('int')

    return(x_train, y_train, x_test, y_test)

#Runs cross validation and outputs fold accuracies, classification report, and the SVM algorithm.
def create_svm(x_train, y_train, x_test, y_test):
    SVM = svm.SVC(C= 100.0, gamma= 1e-06, kernel= 'sigmoid', probability=True)

    cross_pred = cross_val_predict(SVM, x_train, y_train, cv=20, n_jobs=-1)
    accuracy = cross_val_score(SVM, x_train, y_train, cv=20, n_jobs=-1)

    for i in range(20):
        print('Fold {} -- Acc = {}'.format(i, accuracy[i]))

    mean = np.mean(accuracy)
    print(mean)
    
    SVM.fit(x_train, y_train)
    y_pred = SVM.predict(x_test)
    print(classification_report(y_test, y_pred))

    return(SVM)

#Runs cross validation and outputs fold accuracies, classification report, and the logit algorithm.
def create_LR( x_train, y_train, x_test, y_test):
    logit = LogisticRegression(multi_class='auto', solver='liblinear')

    cross_pred = cross_val_predict(logit, x_train, y_train, cv=20, n_jobs=-1)
    acc = cross_val_score(logit, x_train, y_train, cv=20, n_jobs=-1)

    for i in range(20):
        print('Fold {} -- Acc = {}'.format(i, acc[i]))

    mean = np.mean(acc)
    print(mean)

    logit.fit(x_train, y_train)
    y_pred = logit.predict(x_test)
    print(classification_report(y_test, y_pred))

    return(logit)

#Takes both SVM & logit algorithms and also training & testing sets and outputs an ROC graph.
def roc_graph(svm, logit, x_train, y_train, x_test, y_test, title):
    
    y_pred_proba_svm = svm.predict_proba(x_test)[::,1]
    y_pred_proba_log = logit.predict_proba(x_test)[::,1]

    fpr_svm, tpr_svm, _ = metrics.roc_curve(y_test,  y_pred_proba_svm)
    auc_svm = metrics.roc_auc_score(y_test, y_pred_proba_svm)
    auc_svm = round(auc_svm, 2)

    fpr_log, tpr_log, _ = metrics.roc_curve(y_test,  y_pred_proba_log)
    auc_log = metrics.roc_auc_score(y_test, y_pred_proba_log)
    auc_log = round(auc_log, 2)

    plt.plot(fpr_log,tpr_log, label="Logit AUC="+str(auc_log))
    plt.plot(fpr_svm,tpr_svm, label="SVM AUC="+str(auc_svm))
    plt.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC = 0.5)")
    plt.title(f'{title}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()