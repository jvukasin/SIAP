import pandas as pd
import math


def classification_dataset():
    data_combined = pd.read_csv('combined_datasets_w_sunshine.csv')

    df = pd.DataFrame(data_combined, columns=['age', 'country', 'gdp_for_year ($)',
                                              'population', 'salaries', 'sex', 'suicides/100k pop', 'suicides_no',
                                              'sunshine_hours_per_year', 'year'])

    combined_datasets_classification = pd.DataFrame()

    for index, row in df.iterrows():
        row['suicides/100k pop'] = math.ceil(row['suicides/100k pop'] / 5)
        combined_datasets_classification = combined_datasets_classification.append(row)

    combined_datasets_classification.to_csv('combined_datasets_classification.csv')


def split_classification():
    data_combined = pd.read_csv('combined_datasets_classification.csv')

    df = pd.DataFrame(data_combined, columns=['age', 'country', 'gdp_for_year ($)',
                                              'population', 'salaries', 'sex', 'suicides/100k pop', 'suicides_no',
                                              'sunshine_hours_per_year', 'year'])

    master_train = df[df['year'] >= 1990]
    master_train = master_train[master_train['year'] <= 2008]
    # #
    master_test = df[df['year'] >= 2009]
    master_test = master_test[master_test['year'] <= 2016]

    # Izbacivanje zemalja koje postoje u jednom skupu a u drugom ne
    master_train = master_train[master_train['country'] != 'Azerbaijan']
    master_test = master_test[master_test['country'] != 'Bosnia and Herzegovina']
    master_test = master_test[master_test['country'] != 'Turkey']
    # Export podataka

    master_train.to_csv('master_train_classification.csv')
    master_test.to_csv('master_test_classification.csv')

