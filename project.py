import os
import numpy as np
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.ensemble import RandomForestClassifier #RF klasifikator
import xgboost as xgb
import pandas as pd


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


path_master = "/content/drive/My Drive/Colab Notebooks/SIAP/datasets/master.csv"
data_master = pd.read_csv(path_master)

path_salaries = "/content/drive/My Drive/Colab Notebooks/SIAP/datasets/salaries.csv"
data_salaries = pd.read_csv(path_salaries)

df = pd.DataFrame(data_master, columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides/100k pop', 'country-year', 'HDI for year', 'gdp_for_year ($) ', 'gdp_per_capita ($)', 'generation'])

master_train = df[df['year'] >= 1990]
master_train = master_train[master_train['year'] <= 2008]

master_test = df[df['year'] >= 2009]
master_test = master_test[master_test['year'] <= 2016]

print(master_test.shape)
master_train_x = reshape_data(master_train['year'])

"""# Export podataka"""

master_train.to_csv('master_train.csv')
master_test.to_csv('master_test.csv')

files.download('master_test.csv')

"""# RF klasifikator"""

master_train_x = master_train['year'].to_numpy().reshape(-1, 1)
master_train_y = master_train['suicides/100k pop'].to_numpy().reshape(-1,1)

master_test_x = master_test['year'].to_numpy().reshape(-1,1)
master_test_y = master_test['suicides/100k pop'].to_numpy().reshape(-1,1)

data_master_RF = RandomForestClassifier(n_estimators=15)
data_master_RF = data_master_RF.fit(master_train_x, master_train_y)
y_train_pred = data_master_RF.predict(master_train_x)
y_test_pred = data_master_RF.predict(master_test_x)
print("Train accuracy RF: ", accuracy_score(master_train_y, y_train_pred))
print("Validation accuracy RF: ", accuracy_score(master_test_y, y_test_pred))

