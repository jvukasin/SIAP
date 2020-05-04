import pandas as pd
import numpy as np
import math
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

def linear_regression_alg(x_train, y_train, x_test, y_test):
    model = LinearRegression()

    df_x = pd.DataFrame(x_train, columns=['year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year', 'salaries'])
    df_y = pd.DataFrame(y_train, columns=['suicides_no'])

    df_x_test = pd.DataFrame(x_train,
                        columns=['year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year',
                                 'salaries'])
    df_y_test = pd.DataFrame(y_train, columns=['suicides_no'])

    x_train = df_x[['gdp_for_year ($)', 'population', 'salaries', 'sunshine_hours_per_year']].astype(float)
    y_train = df_y[['suicides_no']].astype(float)

    x_test = df_x_test[['gdp_for_year ($)', 'population', 'salaries', 'sunshine_hours_per_year']].astype(float)
    y_test = df_y_test[['suicides_no']].astype(float)

    scores = []

    model.fit(x_train, y_train)
    r_sq = model.score(x_train, y_train)
    # #
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    prediction = model.predict(x_test)

    # print("PREDICTION --")
    # print(prediction)
    #
    # print("Y_TEST -----")
    # y_test = np.array(y_test)
    # print(y_test)

    variance_score = explained_variance_score(y_test, prediction)
    print("Explained variance score Linear Regression: %.2f%%" % (variance_score * 100.0))
