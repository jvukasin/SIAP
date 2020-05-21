import numpy as np


def normalize(df):
    x = df['country']
    # print('x:')
    # print(x)
    result_array = (x - np.min(x)) / (np.max(x) - np.min(x))
    # print('result:')
    # print(result_array)
    # print(result_array[1500])
    df['country'] = result_array
    s = df['age']
    temp = (s - np.min(s)) / (np.max(s) - np.min(s))
    df['age'] = temp
    return df
