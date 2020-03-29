import numpy as np
import pandas as pd
import random_forest_alg as rf

data = pd.read_csv('skup.csv')

df = pd.DataFrame(data, columns=['age', 'country', 'gdp_for_year ($)',
                                        'population', 'salaries', 'sex', 'suicides/100k pop', 'suicides_no', 'year'])

"""# Postoje drzave koje postoje u train skupu a ne postoje u test skupu, i obrutno, pa su one izbacene.
    U train skupu postoji 'Azerbaijan' a ne postoji u test skupu
    U test skupu postoje 'Turkey' i 'Bosnia and Herzegovina' a ne postoje u train skupu
    
    prva = df[df['country'] != 'Azerbaijan']
    druga = prva[prva['country'] != 'Bosnia and Herzegovina']
    treca = druga[druga['country'] != 'Turkey']

    treca.to_csv('skup.csv')
    
    master_train = df[df['year'] >= 1990]
    master_train = master_train[master_train['year'] <= 2008]

    master_test = df[df['year'] >= 2009]
    master_test = master_test[master_test['year'] <= 2016]

    master_train.to_csv('proba_train.csv')
    master_test.to_csv('proba_test.csv')
    
"""

master_train = pd.read_csv('proba_train.csv')
master_test = pd.read_csv('proba_test.csv')

df = pd.DataFrame(master_train, columns=['country', 'year', 'sex', 'age', 'population', 'suicides_no',
                                         'gdp_for_year ($)', 'salaries'])

df_test = pd.DataFrame(master_test, columns=['country', 'year', 'sex', 'age', 'population', 'suicides_no',
                                             'gdp_for_year ($)', 'salaries'])


master_train_x = df[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'salaries']]
master_train_y = df['suicides_no']

master_test_x = df_test[['country', 'year', 'sex', 'age', 'population', 'gdp_for_year ($)', 'salaries']]
master_test_y = df_test['suicides_no']

master_train_x = pd.get_dummies(master_train_x)
master_test_x = pd.get_dummies(master_test_x)

master_train_x.to_csv('POGLEDAJPAOBRISI.csv')

X = np.array(master_train_x)
y = np.array(master_train_y)
X_test = np.array(master_test_x)
y_test = np.array(master_test_y)

# print('Training Features Shape:', X.shape)
# print('Training Labels Shape:', y.shape)
# print('Testing Features Shape:', X_test.shape)
# print('Testing Labels Shape:', y_test.shape)

feature_list = list(master_train_x.columns)

rf.rf_algoritam(X, y, X_test, y_test, feature_list)
