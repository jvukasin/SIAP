import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def linear_regression_alg(x_train, y_train, x_test, y_test):
    model = LinearRegression()

    df = pd.DataFrame(x_train, columns=['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'sunshine_hours_per_year', 'salaries'])

    x_train = np.array(df[['gdp_for_year ($)', 'population', 'salaries', 'sunshine_hours_per_year']])
    # x_test = x_test[['gdp_for_year ($)', 'population', 'salaries', 'sunshine_hours_per_year']]

    print(x_train)
    # print(np.isnan(x_train))

    scores = []
    # model.fit(x_train, y_train)
    # r_sq = model.score(x, y)
    #
    # print('coefficient of determination:', r_sq)
    # print('intercept:', model.intercept_)
    # print('slope:', model.coef_)