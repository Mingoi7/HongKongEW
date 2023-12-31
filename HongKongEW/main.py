import tabula as tb
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import requests
from bs4 import BeautifulSoup
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import jieba
import jieba.analyse
import plotly.graph_objects as go
import scipy.stats as stats

# df = tb.read_pdf(r"C:\Users\User\Mingoi7\HongKongEW\HKPF822 - 2021年各警區的舉報罪案數字和整體罪案率.pdf", encoding="big5" , stream=True , pages='all')

# tb.convert_into(r"C:\Users\User\Mingoi7\HongKongEW\HKPF822 - 2021年各警區的舉報罪案數字和整體罪案率.pdf","test1.csv",pages="1", stream=True , output_format="csv")

# tb.convert_into(r"C:\Users\User\Mingoi7\HongKongEW\HKPF822 - 2021年各警區的舉報罪案數字和整體罪案率.pdf","test2.csv",pages="2", stream=True , output_format="csv")

# tb.convert_into("HKPF818 - 2020年至2021年按罪案類別及警區劃分的舉報案件宗數.pdf","test3.csv",pages="1", stream=True , output_format="csv")

# tb.convert_into("HKPF818 - 2020年至2021年按罪案類別及警區劃分的舉報案件宗數.pdf","test4.csv",pages="2", stream=True , output_format="csv")

# tb.convert_into("HKPF818 - 2020年至2021年按罪案類別及警區劃分的舉報案件宗數.pdf","test5.csv",pages="3", stream=True , output_format="csv")

# tb.convert_into("HKPF818 - 2020年至2021年按罪案類別及警區劃分的舉報案件宗數.pdf","test6.csv",pages="4", stream=True , output_format="csv")
# tb.convert_into("B10100022022MM11B0100.pdf","test7.csv",pages="160", stream=True , output_format="csv")
# tb.convert_into("Hong Kong Property Review Monthly Supplement.pdf","test7.csv",pages="6", stream=True , output_format="csv")

# tb.convert_into("Population and Household Statistics_家庭住戶特徵.pdf","test7.csv",pages="30", stream=True , output_format="csv")
# tb.convert_into("AQR2022c_final.pdf","test7.csv",pages="39", stream=True , output_format="csv")
# tb.convert_into("AQR2022c_final.pdf","test8.csv",pages="40", stream=True , output_format="csv")

csv=pd.read_csv('dataset\property.csv')
studye=pd.read_csv('dataset\studying.csvv')
crime=pd.read_csv('dataset\crime2020.csv')
crime2=pd.read_csv('dataset\crime2021.csv')
population=pd.read_csv('dataset\Population.csv')
pollution1=pd.read_csv('dataset\pollutionEast.csv')
pollution2=pd.read_csv('dataset\pollutionWest.csv')

sns.set_style("whitegrid",{"font.sans-serif":['Microsoft JhengHei']})


# Define two sets of data
crime2020w = crime['Western'].to_numpy()
crime2021w = crime2['Western'].to_numpy()

crime2020e = crime['Eastern'].to_numpy()
crime2021e = crime2['Eastern'].to_numpy()


t_statistic, p_value = stats.ttest_ind(crime2020w, crime2021w)
print("The p-value is:", p_value)

t_statistic, p_value = stats.ttest_ind(crime2020e, crime2021e)
print("The p-value is:", p_value)

def pvalue_101(mu, sigma, samp_size, samp_mean=0, deltam=0):
    np.random.seed(1234)
    s1 = np.random.normal(mu, sigma, samp_size)
    if samp_mean > 0:
        print(len(s1[s1>samp_mean]))
        outliers = float(len(s1[s1>samp_mean])*100)/float(len(s1))
        print('Percentage of numbers larger than {} is {}%'.format(samp_mean, outliers))
    if deltam == 0:
        deltam = abs(mu-samp_mean)
    if deltam > 0 :
        outliers = (float(len(s1[s1>(mu+deltam)]))
                    +float(len(s1[s1<(mu-deltam)])))*100.0/float(len(s1))
        print('Percentage of numbers further than the population mean of {} by +/-{} is {}%'.format(mu, deltam, outliers))

    fig, ax = plt.subplots(figsize=(8,8))
    fig.suptitle('Normal Distribution: population_mean={}'.format(mu) )
    plt.hist(s1)
    plt.axvline(x=mu+deltam, color='red')
    plt.axvline(x=mu-deltam, color='green')
    plt.show()


study1 = len(studye[studye['學校所在分區'].str.contains('東區')])
study2= len(studye[studye['學校所在分區'].str.contains('中西區')])
x_array = np.array([study1,study2])
normalized_arr = preprocessing.normalize([x_array])
print(normalized_arr)

crime1 = crime['Eastern']
crime2= crime['Western']
x_array = np.array([3252,2185])
normalized_arr = preprocessing.normalize([x_array])
print(normalized_arr)

eastern = csv[csv['district'].str.contains('康怡鰂魚涌|太古城|西灣河|筲箕灣柴灣|杏花邨|北角半山|北角')] 
western = csv[csv['district'].str.contains('西半山|堅尼地城西營盤|中半山|山頂')]

eastern_price = (eastern['price'] / eastern['foot']).astype(int)
X_test = eastern_price.iloc[:]
a_test = StandardScaler()
X_result = a_test.fit_transform(np.array(X_test).reshape(len(X_test), 1))

pvalue_101( X_result.mean(), X_result.std(),1000)

eastern = csv[csv['district'].str.contains('康怡鰂魚涌|太古城|西灣河|筲箕灣柴灣|杏花邨|北角半山|北角')] 
western = csv[csv['district'].str.contains('西半山|堅尼地城西營盤|中半山|山頂')]

eastern_price = (eastern['price'] / eastern['foot']).astype(int)
X_test = eastern_price.iloc[:]
a_test = StandardScaler()
X_result = a_test.fit_transform(np.array(X_test).reshape(len(X_test), 1))
print(X_result , '\nCount :',len(X_result),',mean: {} , var:{}'.format( X_result.mean(), X_result.var() ))
sns.displot(X_result,kind='kde')


western_price = (western['price'] / western['foot']).astype(int)
X_test = western_price.iloc[:]
a_test = StandardScaler()
X_result = a_test.fit_transform(np.array(X_test).reshape(len(X_test), 1))
print(X_result ,'\nCount :',len(X_result), ',mean: {} , var:{}'.format( X_result.mean(), X_result.var() ))
sns.displot(X_result,kind='kde')

fig, ax =plt.subplots(1,2,constrained_layout=True, figsize=(16,8))
pic = sns.lineplot(data=pollution1[['二氧化硫', '二氧化氮', '臭氧', '可吸入懸浮粒子PM10','微細懸浮粒子PM2.5']],ax=ax[0])
pic.set_ylim(1,150)
pic.set_title('東區')
pic = sns.lineplot(data=pollution2[['二氧化硫', '二氧化氮', '臭氧', '可吸入懸浮粒子PM10','微細懸浮粒子PM2.5']],ax=ax[1])
pic.set_ylim(1,150)
pic.set_title('中西區')

fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(16,8))
pic = sns.histplot(data=western, x='district',kde=True, ax=ax[0])
pic.set_title('Eastern')
pic = sns.histplot(data=eastern, x='district',ax=ax[1],kde=True)
pic.set_title('Western')
pic = sns.histplot(data=csv, x='district',kde=True, ax=ax[2])
pic.set_title('Total')

fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(16,8))
pic = sns.countplot(x="學校級別", hue="學校所在分區",data=studye, ax=ax[0])
pic.set_title('級別')
pic = sns.countplot(x="資助種類", hue="學校所在分區",data=studye, ax=ax[1])
pic.set_title('資助')
pic = sns.countplot(x="就讀學生性別", hue="學校所在分區", data=studye, ax=ax[2])
pic.set_title('性別')

labels = population.Age.unique()
values = population.Eastern.unique()
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3 , title='Eastern')])
fig.show()

labels = population.Age.unique()
values = population.Western.unique()
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3 , title='Western')])
fig.show()

labels = population.Age.unique()
values = population.Overall.unique()
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3 , title='Overall')])
fig.show()

pic = sns.pairplot(data=eastern, hue='district')
pic = sns.pairplot(data=western, hue='district')
pic = sns.pairplot(data=csv, hue='district')


fig, ax =plt.subplots(1,2,constrained_layout=True, figsize=(16,8))
pic = sns.stripplot(x="district", y="footprice",data=csv, ax=ax[0], hue='district')
pic.set_title('x="district", y="footprice"')
pic = sns.stripplot(x="footprice", y="district", data=csv, ax=ax[1] ,hue='district')
pic.set_title('x="footprice", y="district"')


fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(12, 3))
pic = sns.histplot(csv, x="price", hue="district", ax=ax[0], element="bars")
pic.set_title('element="bars"')
pic = sns.histplot(csv, x="price", hue="district", ax=ax[1], element="step")
pic.set_title('element="step')
pic = sns.histplot(csv, x="price", hue="district", ax=ax[2], element="poly")
pic.set_title('element="poly')

crime['west_rate'] = crime['Western']/crime['Hong Kong Island']
crime['west_rate_total'] = crime['Western']/crime['Total']
crime['east_rate'] = crime['Eastern']/crime['Hong Kong Island']
crime['east_rate_total'] = crime['Eastern']/crime['Total']


plt.figure(figsize=(10,8))
pic = sns.heatmap(crime[['west_rate','east_rate','west_rate_total','east_rate_total']].corr(),annot=True)


plt.figure(figsize=(25,15))
wordcloud = WordCloud(
                        background_color='black',
                        width=1600,
                        height=900,
                        font_path='C:\Windows\Fonts/KAIU.TTF'
                        ).generate(" ".join(crime.Crime))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
