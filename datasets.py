import pandas as pd

def init():
    data_salaries = pd.read_csv('salaries.csv')
    # print(data_salaries)
    df = pd.DataFrame(data_salaries, columns=['1990', '1991', '1992', '1993', '1994',
                                      '1995', '1996', '1997', '1998',
                                      '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
                                      '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
    print(df)
    pom = df.iloc[[6]]
    print(pom)
    pom2 = df.interpolate(method='linear', axis=1, limit=18, inplace=False, limit_direction ='both', downcast=None)
    pom2.to_csv('salaries_interpol.csv')
    print(pom2)

def combine_datasets():
    data_salaries = pd.read_csv('salaries_interpol.csv')
    df_salaries = pd.DataFrame(data_salaries, columns=['country', 'Index', '1990', '1991', '1992', '1993', '1994',
                                              '1995', '1996', '1997', '1998',
                                              '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
                                              '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015',
                                              '2016', '2017'])

    data_master = pd.read_csv('master.csv')
    df_master = pd.DataFrame(data_master, columns=['country', 'year', 'sex', 'age', 'suicides_no', 'population',
                                            'suicides/100k pop', 'country-year', 'gdp_for_year ($)', 'gdp_per_capita ($)',
                                            'generation', 'salaries'])

    combined_datasets = pd.DataFrame()

    # combined_datasets = pd.merge(data_master, data_salaries, on='country', how='outer')
    # print(combined_datasets)

    for index1, row1 in df_master.iterrows():
        for index2, row2 in df_salaries.iterrows():
            if row1['country'] == row2['country']:
                if(row1['year'] == 1990):
                    df_master.at[index1, 'salaries'] = row2['1990']
                    row1['salaries'] = row2['1990']
                elif(row1['year'] == 1991):
                    df_master.at[index1, 'salaries'] = row2['1991']
                    row1['salaries'] = row2['1991']
                elif (row1['year'] == 1992):
                    df_master.at[index1, 'salaries'] = row2['1992']
                    row1['salaries'] = row2['1992']
                elif (row1['year'] == 1993):
                    df_master.at[index1, 'salaries'] = row2['1993']
                    row1['salaries'] = row2['1993']
                elif (row1['year'] == 1994):
                    df_master.at[index1, 'salaries'] = row2['1994']
                    row1['salaries'] = row2['1994']
                elif (row1['year'] == 1995):
                    df_master.at[index1, 'salaries'] = row2['1995']
                    row1['salaries'] = row2['1995']
                elif (row1['year'] == 1996):
                    df_master.at[index1, 'salaries'] = row2['1996']
                    row1['salaries'] = row2['1996']
                elif (row1['year'] == 1997):
                    df_master.at[index1, 'salaries'] = row2['1997']
                    row1['salaries'] = row2['1997']
                elif (row1['year'] == 1998):
                    df_master.at[index1, 'salaries'] = row2['1998']
                    row1['salaries'] = row2['1998']
                elif (row1['year'] == 1999):
                    df_master.at[index1, 'salaries'] = row2['1999']
                    row1['salaries'] = row2['1999']
                elif (row1['year'] == 2000):
                    df_master.at[index1, 'salaries'] = row2['2000']
                    row1['salaries'] = row2['2000']
                elif (row1['year'] == 2001):
                    df_master.at[index1, 'salaries'] = row2['2001']
                    row1['salaries'] = row2['2001']
                elif (row1['year'] == 2002):
                    df_master.at[index1, 'salaries'] = row2['2002']
                    row1['salaries'] = row2['2002']
                elif (row1['year'] == 2003):
                    df_master.at[index1, 'salaries'] = row2['2003']
                    row1['salaries'] = row2['2003']
                elif (row1['year'] == 2004):
                    df_master.at[index1, 'salaries'] = row2['2004']
                    row1['salaries'] = row2['2004']
                elif (row1['year'] == 2005):
                    df_master.at[index1, 'salaries'] = row2['2005']
                    row1['salaries'] = row2['2005']
                elif (row1['year'] == 2006):
                    df_master.at[index1, 'salaries'] = row2['2006']
                    row1['salaries'] = row2['2006']
                elif (row1['year'] == 2007):
                    df_master.at[index1, 'salaries'] = row2['2007']
                    row1['salaries'] = row2['2007']
                elif (row1['year'] == 2008):
                    df_master.at[index1, 'salaries'] = row2['2008']
                    row1['salaries'] = row2['2008']
                elif (row1['year'] == 2009):
                    df_master.at[index1, 'salaries'] = row2['2009']
                    row1['salaries'] = row2['2009']
                elif (row1['year'] == 2010):
                    df_master.at[index1, 'salaries'] = row2['2010']
                    row1['salaries'] = row2['2010']
                elif (row1['year'] == 2011):
                    df_master.at[index1, 'salaries'] = row2['2011']
                    row1['salaries'] = row2['2011']
                elif (row1['year'] == 2012):
                    df_master.at[index1, 'salaries'] = row2['2012']
                    row1['salaries'] = row2['2012']
                elif (row1['year'] == 2013):
                    df_master.at[index1, 'salaries'] = row2['2013']
                    row1['salaries'] = row2['2013']
                elif (row1['year'] == 2014):
                    df_master.at[index1, 'salaries'] = row2['2014']
                    row1['salaries'] = row2['2014']
                elif (row1['year'] == 2015):
                    df_master.at[index1, 'salaries'] = row2['2015']
                    row1['salaries'] = row2['2015']
                elif (row1['year'] == 2016):
                    df_master.at[index1, 'salaries'] = row2['2016']
                    row1['salaries'] = row2['2016']
                combined_datasets = combined_datasets.append(row1)

    combined_datasets.to_csv('combined_datasets.csv')
    df_master.to_csv('combined_datasets2.csv')





