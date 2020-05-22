

def string_to_int_columns(df):
    countries = df['country']
    finalC = []

    for i in countries:
        if i not in finalC:
            finalC.append(i)
        # if i == 'Azerbaijan':
        #     print("Azerbaijan: ", finalC.index(i) + 1)
        # if i == 'Bosnia and Herzegovina':
        #     print("Bosnia and Herzegovina: ", finalC.index(i) + 1)
        # if i == 'Turkey':
        #     print("Turkey: ", finalC.index(i) + 1)



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
