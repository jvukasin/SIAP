import os
import numpy as np
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
import random_forest_alg as rf
import xgboost_alg as xgb
from sklearn import preprocessing
import structure_colum_string_values as scsv
import pandas as pd


# datasets.init()
# datasets.combine_datasets()

# def reshape_data(input_data):
#     nsamples, nx, ny = input_data.shape
#     return input_data.reshape((nsamples, nx*ny))
#
# # print(data_master['gdp_for_year ($)'])

data_combined = pd.read_csv('combined_datasets.csv')
master_train = pd.read_csv('master_train.csv')
master_test = pd.read_csv('master_test.csv')
df = pd.DataFrame(data_combined, columns=['age', 'country', 'gdp_for_year ($)',
                                        'population', 'salaries', 'sex', 'suicides/100k pop', 'suicides_no', 'year'])


"""# Prebacivanje stringova iz kolona u intove zbog algoritama"""
# scsv.string_to_int_columns(df)

"""# PRAVLJENJE TRAIN I TEST SKUPA NA OSNOVU GODINA"""
# master_train = df[df['year'] >= 1990]
# master_train = master_train[master_train['year'] <= 2008]
#
# master_test = df[df['year'] >= 2009]
# master_test = master_test[master_test['year'] <= 2016]
#
# Export podataka
#
# master_train.to_csv('master_train.csv')
# master_test.to_csv('master_test.csv')


"""# RF klasifikator"""

df = pd.DataFrame(master_train, columns=['country', 'year', 'sex', 'age', 'population', 'suicides_no',
                                         'gdp_for_year ($)']) # + salaries

df_test = pd.DataFrame(master_test, columns=['country', 'year', 'sex', 'age', 'population', 'suicides_no',
                                             'gdp_for_year ($)']) # + salaries

master_train_x = df[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)']]
master_train_y = df['suicides_no']

master_test_x = df_test[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)']]
master_test_y = df_test['suicides_no']

# x = pd.concat([master_train_x, master_test_x])
# y = pd.concat([master_train_y, master_test_y])

# master_train_x = pd.get_dummies(master_train_x) # get dummies stavlja true false za stringove, pogledati fajl POGLEDAJPAOBRISI

X = np.array(master_train_x)
y = np.array(master_train_y)

X_test = np.array(master_test_x)
y_test = np.array(master_test_y)

y_train = (y * 100).astype(int)
y_test = (y_test * 100).astype(int)

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)

new_y = y * 100
new_y = new_y.astype(int)

new_y_test = y_test * 100
new_y_test = new_y_test.astype(int)

lab_enc = preprocessing.LabelEncoder()
encodedTEST = lab_enc.fit_transform(y_test)

"""# RandomForest algoritam"""
rf.rf_algoritam(X, y_train, X_test, y_test)

"""# XGBoost algoritam"""
xgb.xbg_algoraim(X, y_train, X_test, y_test)
