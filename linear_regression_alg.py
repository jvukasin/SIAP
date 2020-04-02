import pandas as pd
import numpy as np
import math
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def linear_regression_alg(x_train, y_train, x_test, y_test):
    model = LinearRegression()

    df_x = pd.DataFrame(x_train, columns=['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year', 'salaries'])
    df_y = pd.DataFrame(y_train, columns=['suicides_no'])
    x_train = df_x[['gdp_for_year ($)', 'population', 'salaries', 'sunshine_hours_per_year']].astype(float)
    y_train = df_y[['suicides_no']].astype(float)
    # x_test = x_test[['gdp_for_year ($)', 'population', 'salaries', 'sunshine_hours_per_year']]
    x_train_f = pd.DataFrame()
    y_train_f =  pd.DataFrame()



    scores = []
    model.fit(x_train, y_train)
    r_sq = model.score(x_train, y_train)
    # #
    print('coefficient of determination:', r_sq)
    # print('intercept:', model.intercept_)
    # print('slope:', model.coef_)
