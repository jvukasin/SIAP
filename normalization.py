import numpy as np


def normalize(df):
    x = df['country']
    result_array = (x - np.min(x)) / (np.max(x) - np.min(x))
    df['country'] = result_array

    s = df['age']
    temp = (s - np.min(s)) / (np.max(s) - np.min(s))
    df['age'] = temp

    # s = df['year']
    # temp = (s - np.min(s)) / (np.max(s) - np.min(s))
    # df['year'] = temp

    s = df['population']
    temp = (s - np.min(s)) / (np.max(s) - np.min(s))
    df['population'] = temp

    s = df['gdp_for_year ($)']
    temp = (s - np.min(s)) / (np.max(s) - np.min(s))
    df['gdp_for_year ($)'] = temp

    s = df['sunshine_hours_per_year']
    temp = (s - np.min(s)) / (np.max(s) - np.min(s))
    df['sunshine_hours_per_year'] = temp

    s = df['salaries']
    temp = (s - np.min(s)) / (np.max(s) - np.min(s))
    df['salaries'] = temp

    # df = (df - np.min(df)) / (np.max(df) - np.min(df))
    return df
