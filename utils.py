import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import log_loss, confusion_matrix, precision_score, recall_score, f1_score

def split_column_by_type(df, column_labels):
    categorical_column_labels = []
    numerical_column_labels = []
    
    for column_label in column_labels:
        if (df.dtypes[column_label] == 'object'):
            categorical_column_labels.append(column_label)
        else:
            numerical_column_labels.append(column_label)
    return categorical_column_labels, numerical_column_labels

def transform_column_to_onehot(df, selected_columns, dict_encoder ):
    X = np.array([])
    for column_label in selected_columns:
        col = df[column_label].to_numpy().reshape(-1, 1)
        if (df.dtypes[column_label] == 'object'):
            enc = OneHotEncoder(handle_unknown='ignore')
            enc = enc.fit(col)
            dict_encoder[column_label] = enc
            col = enc.transform(col).toarray()
        if len(X) == 0:
            X = col
        else:
            X = np.concatenate((X, col), axis=1)
    return X, dict_encoder

def label_output_feature(y):
    le = preprocessing.LabelEncoder()
    le.fit(y)
    yhat = le.transform(y)
    return yhat, le

def split_data_and_train_model(X, y, train_split_ratio):
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split_ratio / 100.0, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def transform_data_to_predict(selected_columns, dict_data_to_prediction):
    data_to_prediction = np.array([])
    for data_column in selected_columns:
        data_to_prediction = np.append(data_to_prediction, dict_data_to_prediction[data_column])
    data = np.array(data_to_prediction).reshape(1, -1)
    return data

def score_model(model: LogisticRegression, X_test, y_test):
    score = model.score(X_test, y_test)
    return score

def get_precision(model, X_test, y_test):
    prediction = model.predict(X_test)
    score = precision_score(y_test, prediction, average='micro')
    return score

def get_recall(model, X_test, y_test):
    prediction = model.predict(X_test)
    score = recall_score(y_test, prediction, average='micro')
    return score

def get_f1(model, X_test, y_test):
    prediction = model.predict(X_test)
    score = f1_score(y_test, prediction, average='micro')
    return score

def get_logloss(model, X_test, y_test):
    prediction = model.predict_proba(X_test)
    print(prediction.shape)
    score = log_loss(y_test, prediction)
    return score

def train_with_kFold(X, y, n_splits):
    model = LogisticRegression()
    kf = KFold(n_splits=n_splits)
    train_score = []
    test_score = []
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        score_test = score_model(model, X_test, y_test)
        score_train = score_model(model, X_train, y_train)
        train_score.append(score_train)
        test_score.append(score_test)
    return model, train_score, test_score

def get_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cfs_matrix = confusion_matrix(y_test, y_pred)
    return cfs_matrix