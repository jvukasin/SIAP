import pandas as pd
import numpy as np
import math
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV


def linear_regression_alg(x_train, y_train, x_test, y_test):
    model = LinearRegression(normalize=True)
    model.fit(x_train, y_train)

    # MSEs = cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    # mean_MSE = np.mean(MSEs)
    # print('mean:', mean_MSE)

    ridge_001 = Ridge(alpha=0.01)
    ridge_001.fit(x_train, y_train)

    ridge_100 = Ridge(alpha=100)
    ridge_100.fit(x_train, y_train)

    # ridge = Ridge()
    # parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    # ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
    # ridge_regressor.fit(x_train, y_train)
    # print('best param ridge: ', ridge_regressor.best_params_)
    # print('best score ridge: ', ridge_regressor.best_score_)

    # lasso = Lasso()
    # parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    # lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
    # lasso_regressor.fit(x_train, y_train)
    # print('best param lasso: ', lasso_regressor.best_params_)
    # print('best score lasso: ', lasso_regressor.best_score_)

    lasso = Lasso()
    lasso.fit(x_train, y_train)
    train_score = lasso.score(x_train, y_train)
    test_score = lasso.score(x_test, y_test)
    coeff_used = np.sum(lasso.coef_ != 0)
    # print('lasso train score:', train_score)
    print('lasso test score:', test_score)
    print('coeff used: ', coeff_used)

    lasso001 = Lasso(alpha=0.01, max_iter=10e5)
    lasso001.fit(x_train, y_train)
    train_score001 = lasso001.score(x_train, y_train)
    test_score001 = lasso001.score(x_test, y_test)
    coeff_used001 = np.sum(lasso001.coef_ != 0)
    # print('lasso001 train score:', train_score001)
    print('lasso001 test score:', test_score001)
    print('coeff used: ', coeff_used001)

    lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
    lasso00001.fit(x_train, y_train)
    train_score00001 = lasso00001.score(x_train, y_train)
    test_score00001 = lasso00001.score(x_test, y_test)
    coeff_used00001 = np.sum(lasso00001.coef_ != 0)
    # print('lasso00001 train score:', train_score00001)
    print('lasso00001 test score:', test_score00001)
    print('coeff used: ', coeff_used00001)

    # df_x = pd.DataFrame(x_train, columns=['year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year', 'salaries'])
    # df_y = pd.DataFrame(y_train, columns=['suicides_no'])
    #
    # df_x_test = pd.DataFrame(x_test,
    #                     columns=['year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year',
    #                              'salaries'])
    # df_y_test = pd.DataFrame(y_test, columns=['suicides_no'])
    #
    # x_train = df_x[['gdp_for_year ($)', 'population', 'salaries', 'sunshine_hours_per_year']].astype(float)
    # y_train = df_y[['suicides_no']].astype(float)
    #
    # x_test = df_x_test[['gdp_for_year ($)', 'population', 'salaries', 'sunshine_hours_per_year']].astype(float)
    # y_test = df_y_test[['suicides_no']].astype(float)

    scores = []

    r_sq = model.score(x_train, y_train)
    model_test_score = model.score(x_test, y_test)
    # print('model train score:', r_sq)
    print('model test score:', model_test_score)

    ridge_001_score_train = ridge_001.score(x_train, y_train)
    ridge_001_score_test = ridge_001.score(x_test, y_test)
    # print('ridge_001 train score:', ridge_001_score_train)
    print('ridge_001 test score:', ridge_001_score_test)

    ridge_100_score_train = ridge_100.score(x_train, y_train)
    ridge_100_score_test = ridge_100.score(x_test, y_test)
    # print('ridge_100 train score:', ridge_100_score_train)
    print('ridge_100 test score:', ridge_100_score_test)
    # #
    # print('coefficient of determination:', r_sq)
    # print('intercept:', model.intercept_)
    # print('slope:', model.coef_)

    prediction = model.predict(x_test)

    # print("PREDICTION --")
    # print(prediction)
    #
    # print("Y_TEST -----")
    # y_test = np.array(y_test)
    # print(y_test)

    variance_score = explained_variance_score(y_test, prediction)
    print("Explained variance score Linear Regression: %.2f%%" % (variance_score * 100.0))

    r2 = r2_score(y_test, prediction)
    print("R^2 score Linear Regression: %.2f" % r2)

    mean = mean_squared_error(y_test, prediction)
    print("Mean squared error Linear Regression: %.2f" % mean)

    rmse = np.sqrt(mean)
    print("Root mean square error Linear Regression: %.2f" % rmse)
