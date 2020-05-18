import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
import random_forest_alg as rf
import xgboost_alg as xgb
import pca_alg as pca
import linear_regression_alg as lr
import gradient_boosted_tree_alg as gbt
import svr_alg as svr
from sklearn import preprocessing
import pandas as pd
import label_enc_try as lt


# datasets.init()
# datasets.combine_datasets()

# def reshape_data(input_data):
#     nsamples, nx, ny = input_data.shape
#     return input_data.reshape((nsamples, nx*ny))
#
# # print(data_master['gdp_for_year ($)'])

data_combined = pd.read_csv('combined_datasets_w_sunshine.csv')
master_train = pd.read_csv('master_train.csv')
master_test = pd.read_csv('master_test.csv')
master_train_classification = pd.read_csv('master_train_classification.csv')
master_test_classification = pd.read_csv('master_test_classification.csv')
df = pd.DataFrame(data_combined, columns=['age', 'country', 'gdp_for_year ($)',
                                        'population', 'salaries', 'sex', 'suicides/100k pop', 'suicides_no', 'sunshine_hours_per_year', 'year'])

whole_dataset = df

"""Label encoder"""
lt.try_all_algs_with_le(whole_dataset)

"""# PRAVLJENJE TRAIN I TEST SKUPA NA OSNOVU GODINA"""
# master_train = df[df['year'] >= 1990]
# master_train = master_train[master_train['year'] <= 2008]
# # #
# master_test = df[df['year'] >= 2009]
# master_test = master_test[master_test['year'] <= 2016]
#
# # Izbacivanje zemalja koje postoje u jednom skupu a u drugom ne
# master_train = master_train[master_train['country'] != 'Azerbaijan']
# master_test = master_test[master_test['country'] != 'Bosnia and Herzegovina']
# master_test = master_test[master_test['country'] != 'Turkey']
# # Export podataka
#
# master_train.to_csv('master_train.csv')
# master_test.to_csv('master_test.csv')

df = pd.DataFrame(master_train, columns=['country', 'year', 'sex', 'age', 'population', 'suicides_no',
                                         'gdp_for_year ($)', 'sunshine_hours_per_year', 'salaries'])

df_test = pd.DataFrame(master_test, columns=['country', 'year', 'sex', 'age', 'population', 'suicides_no',
                                             'gdp_for_year ($)', 'sunshine_hours_per_year', 'salaries'])


master_train_x = df[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year',
                     'salaries']]
master_train_y = df['suicides_no']

master_test_x = df_test[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year',
                         'salaries']]
master_test_y = df_test['suicides_no']


master_train_x = pd.get_dummies(master_train_x)
master_test_x = pd.get_dummies(master_test_x)

X = np.array(master_train_x)
y = np.array(master_train_y)

X_test = np.array(master_test_x)
y_test = np.array(master_test_y)

features = master_train_x.columns

"""# RandomForest algoritam"""
# lab_enc = preprocessing.LabelEncoder()
# print('pre')
# print(y)
# y = lab_enc.fit_transform(y)
# print('posle')
# print(y)
# y_test = lab_enc.fit_transform(y_test)

print('*****************REGRESSION*****************')

print('====================RF====================')
# rf.rf_algoritam(X, y, X_test, y_test, features)

"""# XGBoost algoritam"""
print('==================XGBoost=================')
# xgb.xbg_algoritam(X, y, X_test, y_test)

"""# PCA algoritam"""
print('====================PCA===================')
# pca.pca_algoritam(X, y, X_test, y_test)

"""# Linear regression algoritam"""
print('=================Linear Reg===============')
# lr.linear_regression_alg(master_train_x, master_train_y, master_test_x, master_test_y)

"""# Gradient Boosted Tree"""
print('====================GBT===================')
# gbt.gbt_algorythm(X, y, X_test, y_test)

"""# Support Vector Regression (SVR)"""
print('====================SVR===================')
# svr.svr_algorithm(X, y, X_test, y_test)


print('*****************CLASSIFICATION*****************')

# df_train_classification = pd.DataFrame(master_train_classification, columns=['country', 'year', 'sex', 'age',
#                                                                              'population', 'suicides/100k pop',
#                                                                              'gdp_for_year ($)',
#                                                                              'sunshine_hours_per_year', 'salaries'])
#
# df_test_classification = pd.DataFrame(master_test_classification, columns=['country', 'year', 'sex', 'age', 'population',
#                                                                            'suicides/100k pop', 'gdp_for_year ($)',
#                                                                            'sunshine_hours_per_year', 'salaries'])
#
#
# master_train_x = df_train_classification[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)',
#                                           'sunshine_hours_per_year', 'salaries']]
# master_train_y = df_train_classification['suicides/100k pop']
#
# master_test_x = df_test_classification[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)',
#                                         'sunshine_hours_per_year', 'salaries']]
# master_test_y = df_test_classification['suicides/100k pop']
#
#
# master_train_x = pd.get_dummies(master_train_x)
# master_test_x = pd.get_dummies(master_test_x)
#
# X = np.array(master_train_x)
# y = np.array(master_train_y)
#
# X_test = np.array(master_test_x)
# y_test = np.array(master_test_y)
#
# features = master_train_x.columns
#
# print('====================RF====================')
# rf.rf_algoritam(X, y, X_test, y_test, features)
