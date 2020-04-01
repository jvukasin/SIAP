import pandas as pd
from statistics import median
import numpy as np
import csv


def extract_contries():
    country_names = pd.read_csv('country_names.csv')
    hours = pd.read_csv('hours.csv')

    df_country_names = pd.DataFrame(country_names, columns=['0'])
    df_hours = pd.DataFrame(hours, columns=['0'])

    combined_datasets = pd.concat([df_country_names, df_hours], axis=1, join='outer')

    pd.DataFrame(combined_datasets).to_csv("combined_sunshine.csv")


def merge_cities_for_one_country():
    sunshine_raw = pd.read_csv('combined_sunshine.csv')
    current_country = ""
    count = 0
    sum_hours = 0
    one_country_hours = []
    final_sunshine = pd.DataFrame(columns=['country', 'hours'])

    for index, row in sunshine_raw.iterrows():
        if index == 0:
            current_country = row['country']

        if current_country == row['country']:
            count += 1
            hours = float(row['hours'].replace(",", ""))
            one_country_hours.append(hours)
            sum_hours += hours

        else:
            average = sum_hours/count

            if count > 4:
                median_value = median(one_country_hours)
            else:
                median_value = average

            # printing
            print("Current country: ", current_country)
            print("Count: ", count)
            print("Total hours: ", sum_hours)
            print("List: ", one_country_hours)
            print("Average: ", average)
            print("Median: ", median_value)

            print()

            # insert into final sunshine dataframe
            data_to_append = {'country': current_country, 'hours': median_value}
            print(data_to_append)
            final_sunshine = final_sunshine.append(data_to_append, ignore_index=True, sort=False)

            current_country = row['country']
            count = 1
            hours = float(row['hours'].replace(",", ""))
            one_country_hours = [hours]
            sum_hours = hours

    print(final_sunshine)
    final_sunshine.to_csv("combined_sunshine_median.csv")


def insert_in_master():

    sunshine = pd.read_csv('combined_sunshine_median.csv')
    master = pd.read_csv('combined_datasets.csv')

    new_master = pd.DataFrame()

    for index1, row1 in master.iterrows():
        for index2, row2 in sunshine.iterrows():
            if row1['country'] == row2['country']:
                row1['sunshine_hours_per_year'] = row2['hours']

        new_master = new_master.append(row1)

    new_master.to_csv('combined_datasets_w_sunshine.csv')


insert_in_master()

