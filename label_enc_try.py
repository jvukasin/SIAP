from sklearn import preprocessing
import numpy as np
import random_forest_alg as rf
import xgboost_alg as xgb
import svr_alg as svr
import normalization
import pca_alg as pca
import linear_regression_alg as lr
import gradient_boosted_tree_alg as gbt


def try_all_algs_with_le(df):

    le = preprocessing.LabelEncoder()
    df['country'] = le.fit_transform(df['country'])
    df['sex'] = le.fit_transform(df['sex'])
    df['age'] = le.fit_transform(df['age'])

    df = df[df['country'] != 'Azerbaijan']
    df = df[df['country'] != 'Bosnia and Herzegovina']
    df = df[df['country'] != 'Turkey']

    # df = normalization.normalize(df)

    master_train = df[df['year'] >= 1990]
    master_train = master_train[master_train['year'] <= 2008]

    master_test = df[df['year'] >= 2009]
    master_test = master_test[master_test['year'] <= 2016]

    # Izbacivanje zemalja koje postoje u jednom skupu a u drugom ne
    # master_train = master_train[master_train['country'] != 'Azerbaijan']
    # master_test = master_test[master_test['country'] != 'Bosnia and Herzegovina']
    # master_test = master_test[master_test['country'] != 'Turkey']

    master_train_x = master_train[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year',
                                   'salaries']]
    master_train_y = master_train['suicides_no']

    master_test_x = master_test[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year',
                                 'salaries']]
    master_test_y = master_test['suicides_no']

    X = np.array(master_train_x)
    y = np.array(master_train_y)

    X_test = np.array(master_test_x)
    y_test = np.array(master_test_y)

    features = master_train_x.columns

    print('====================RF====================')
    # rf.rf_algoritam(X, y, X_test, y_test, features)

    """# XGBoost algoritam"""
    print('==================XGBoost=================')
    # xgb.xbg_algoritam(X, y, X_test, y_test)

    """# PCA algoritam"""
    print('====================PCA===================')
    # pca.pca_algoritam(X, y, X_test, y_test)

    """# Gradient Boosted Tree"""
    print('====================GBT===================')
    gbt.gbt_algorythm(X, y, X_test, y_test)

    """# Support Vector Regression (SVR)"""
    print('====================SVR===================')
    # svr.svr_algorithm(X, y, X_test, y_test)






