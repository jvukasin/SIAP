import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class AgeGroup:
    name: ""
    maleSum: 0
    maleCnt: 0
    femaleSum: 0
    femaleCnt: 0
    malePc: 0
    femalePc: 0

    def __init__(self, name, maleSum, femaleSum):
        self.name = name
        self.maleCnt = 0
        self.maleSum = maleSum
        self.malePc = 0
        self.femaleCnt = 0
        self.femaleSum = femaleSum
        self.femalePc = 0


def plots(dataset: pd.DataFrame):
    # Male / female percentage ratio by age group
    [g_5_14, g_15_24, g_25_34, g_35_54, g_55_74, g_75] = aggregate_data_by_age(dataset)
    plot_by_age_and_sex(g_5_14, g_15_24, g_25_34, g_35_54, g_55_74, g_75)

    # By country's GDP
    [gdp, suicides] = aggregate_data_gdp(dataset)
    plot_by_gdp(gdp, suicides)

    # By no of suicide by country
    plot_suicides_total_per100k(dataset)


def aggregate_data_by_age(dataset):
    g_5_14 = AgeGroup("5-14", 0, 0)
    g_15_24 = AgeGroup("15-24", 0, 0)
    g_25_34 = AgeGroup("25-34", 0, 0)
    g_35_54 = AgeGroup("35-54", 0, 0)
    g_55_74 = AgeGroup("55-74", 0, 0)
    g_75 = AgeGroup("75+", 0, 0)

    for index, row in dataset.iterrows():

        if index % 1000 == 0:
            print("working...")

        if row['age'] == "5-14 years":
            if row['sex'] == "male":
                g_5_14.maleSum += row['suicides_no']
                g_5_14.maleCnt += 1
            elif row['sex'] == "female":
                g_5_14.femaleSum += row['suicides_no']
                g_5_14.femaleCnt += 1

            continue

        if row['age'] == "15-24 years":
            if row['sex'] == "male":
                g_15_24.maleSum += row['suicides_no']
                g_15_24.maleCnt += 1
            elif row['sex'] == "female":
                g_15_24.femaleSum += row['suicides_no']
                g_15_24.femaleCnt += 1

            continue

        if row['age'] == "25-34 years":
            if row['sex'] == "male":
                g_25_34.maleSum += row['suicides_no']
                g_25_34.maleCnt += 1
            elif row['sex'] == "female":
                g_25_34.femaleSum += row['suicides_no']
                g_25_34.femaleCnt += 1

            continue

        if row['age'] == "35-54 years":
            if row['sex'] == "male":
                g_35_54.maleSum += row['suicides_no']
                g_35_54.maleCnt += 1
            elif row['sex'] == "female":
                g_35_54.femaleSum += row['suicides_no']
                g_35_54.femaleCnt += 1

            continue

        if row['age'] == "55-74 years":
            if row['sex'] == "male":
                g_55_74.maleSum += row['suicides_no']
                g_55_74.maleCnt += 1
            elif row['sex'] == "female":
                g_55_74.femaleSum += row['suicides_no']
                g_55_74.femaleCnt += 1

            continue

        if row['age'] == "75+ years":
            if row['sex'] == "male":
                g_75.maleSum += row['suicides_no']
                g_75.maleCnt += 1
            elif row['sex'] == "female":
                g_75.femaleSum += row['suicides_no']
                g_75.femaleCnt += 1

    # Normalize data
    [g_5_14.malePc, g_5_14.femalePc] = calculate_percentage(g_5_14.maleSum, g_5_14.femaleSum)
    [g_15_24.malePc, g_15_24.femalePc] = calculate_percentage(g_15_24.maleSum, g_15_24.femaleSum)
    [g_25_34.malePc, g_25_34.femalePc] = calculate_percentage(g_25_34.maleSum, g_25_34.femaleSum)
    [g_35_54.malePc, g_35_54.femalePc] = calculate_percentage(g_35_54.maleSum, g_35_54.femaleSum)
    [g_55_74.malePc, g_55_74.femalePc] = calculate_percentage(g_55_74.maleSum, g_55_74.femaleSum)
    [g_75.malePc, g_75.femalePc] = calculate_percentage(g_75.maleSum, g_75.femaleSum)

    return [g_5_14, g_15_24, g_25_34, g_35_54, g_55_74, g_75]


def aggregate_data_gdp(dataset):
    gdp = []
    suicides = []

    for index, row in dataset.iterrows():
        if index % 1000 == 0:
            print("working...")

        if row["suicides/100k pop"] == 0:
            continue
        else:
            gdp.append(row['gdp_per_capita ($)'])
            suicides.append(row['suicides/100k pop'])

    return [gdp, suicides]


def calculate_percentage(x, y):
    xPc = 0
    yPc = 0
    if y >= x:
        xPc = (x / y) * 100
        yPc = 100 - xPc
    else:
        yPc = (y / x) * 100
        xPc = 100 - yPc

    return [xPc, yPc]


def plot_by_age_and_sex(g1: AgeGroup, g2: AgeGroup, g3: AgeGroup, g4: AgeGroup, g5: AgeGroup, g6: AgeGroup):

    N = 6
    menMeans = (g1.malePc, g2.malePc, g3.malePc, g4.malePc, g5.malePc, g6.malePc)
    womenMeans = (g1.femalePc, g2.femalePc, g3.femalePc, g4.femalePc, g5.femalePc, g6.femalePc)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, menMeans, width,)
    p2 = plt.bar(ind, womenMeans, width, bottom=menMeans,)

    plt.ylabel('Percentage [%]')
    plt.title('Male / female suicide number ratio by age group')
    plt.xticks(ind, ('5-14 y', '15-24 y', '25-34 y', '35-54 y', '55-74 y', '75+ y'))
    plt.legend((p1[0], p2[0]), ('Men', 'Women'))

    plt.show()


def plot_by_gdp(gdp, suicides):
    plt.scatter(gdp, suicides, alpha=0.9, marker='.', cmap='viridis')
    plt.title("Global relationship between GDP and suicides")
    plt.xlabel("GDP per capita ($)")
    plt.ylabel("Suicides 100k / pop")
    plt.show()


def plot_suicides_total_per100k(dataset):
    new_dataset = dataset[dataset['year'] >= 1990]

    suicide_no = []
    countries = []
    country1 = "Albania"
    sum = 0

    for index, row in new_dataset.iterrows():
        if index % 1000 == 0:
            print("working...")

        if row["suicides/100k pop"] == 0:
            continue
        else:
            country2 = row['country']
            if country2 == country1:
                sum = sum + row['suicides/100k pop']
            else:
                suicide_no.append(sum)
                countries.append(country1)
                country1 = country2
                sum = 0
                sum = sum + row['suicides/100k pop']

    df = pd.DataFrame({"country": countries,
                       "suicide_no": suicide_no})
    df_sort = df.sort_values('suicide_no')
    coun = df_sort['country']
    suic = df_sort['suicide_no']

    y_pos = np.arange(len(coun))
    plt.barh(y_pos, suic, align='center')
    plt.title("Number of suicides per 100k population by country from 1990 to 2016")
    plt.yticks(y_pos, coun)
    plt.xlabel("Number of suicides per 100.000 people")
    plt.ylabel("Country")
    plt.show()
