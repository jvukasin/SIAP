import numpy as np


def normalize(df):
    x = df['country']
    result_array = (x - np.min(x)) / (np.max(x) - np.min(x))
    df['country'] = result_array
    s = df['age']
    temp = (s - np.min(s)) / (np.max(s) - np.min(s))
    df['age'] = temp
    return df
