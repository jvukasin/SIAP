import os
import numpy as np
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.ensemble import RandomForestClassifier #RF klasifikator
from sklearn import preprocessing
import xgboost as xgb
import pandas as pd
import datasets


# datasets.init()
# datasets.combine_datasets()

# def reshape_data(input_data):
#     nsamples, nx, ny = input_data.shape
#     return input_data.reshape((nsamples, nx*ny))
#
#
data_combined = pd.read_csv('combined_datasets.csv')
#
#
# # print(data_master['gdp_for_year ($)'])
#
df = pd.DataFrame(data_combined, columns=['age', 'country', 'gdp_for_year ($)',
                                        'population', 'salaries', 'sex', 'suicides/100k pop', 'year'])

countries = df['country']
finalC = []

for i in countries:
    if i not in finalC:
        finalC.append(i)

print(finalC)
print(len(finalC))

for index1, row1 in df.iterrows():
    for contr in finalC:
        if row1['country'] == contr:
            df.at[index1, 'country'] = finalC.index(contr) + 1
            break

for index1, row1 in df.iterrows():
    if row1['sex'] == 'male':
        df.at[index1, 'sex'] = 0
        row1['sex'] = 0
    elif row1['sex'] == 'female':
        df.at[index1, 'sex'] = 1
        row1['sex'] = 1

for index1, row1 in df.iterrows():
    if row1['age'] == '5-14 years':
        df.at[index1, 'age'] = 1
    elif row1['age'] == '15-24 years':
        df.at[index1, 'age'] = 2
    elif row1['age'] == '25-34 years':
        df.at[index1, 'age'] = 3
    elif row1['age'] == '35-54 years':
        df.at[index1, 'age'] = 4
    elif row1['age'] == '55-74 years':
        df.at[index1, 'age'] = 5
    elif row1['age'] == '75+ years':
        df.at[index1, 'age'] = 6

# PRAVLJENJE TRAIN I TEST SKUPA NA OSNOVU GODINA

master_train = df[df['year'] >= 1990]
master_train = master_train[master_train['year'] <= 2008]

master_test = df[df['year'] >= 2009]
master_test = master_test[master_test['year'] <= 2016]


"""# Export podataka"""

master_train.to_csv('master_train.csv')
master_test.to_csv('master_test.csv')

"""# RF klasifikator"""

# df = pd.DataFrame(master_train, columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides/100k pop',
#                                  'HDI for year', 'gdp_per_capita ($)'])
#
# df_test = pd.DataFrame(master_test, columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides/100k pop',
#                                  'HDI for year', 'gdp_per_capita ($)'])
#
# master_train_x = df[['year', 'population', 'gdp_per_capita ($)']]
# master_train_y = df['suicides/100k pop']
#
# master_test_x = df_test[['year', 'population', 'gdp_per_capita ($)']]
# master_test_y = df_test['suicides/100k pop']
#
#
# X = np.array(master_train_x)
# y = np.array(master_train_y)
#
# lab_enc = preprocessing.LabelEncoder()
# training_scores_encoded = lab_enc.fit_transform(y)
#
# X_test = np.array(master_test_x)
# y_test = np.array(master_test_y)
#
# training_scores_encoded_test = lab_enc.fit_transform(y_test)
#
# data_master_RF = RandomForestClassifier(50)
# data_master_RF = data_master_RF.fit(X, training_scores_encoded)
# y_train_pred = data_master_RF.predict(X)
# y_test_pred = data_master_RF.predict(X_test)
# print("Train accuracy RF: ", accuracy_score(training_scores_encoded, y_train_pred))
# print("Validation accuracy RF: ", accuracy_score(training_scores_encoded_test, y_test_pred))

