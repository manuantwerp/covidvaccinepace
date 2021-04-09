# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:23:07 2021

@author: manue
"""

import numpy as np  # useful for many scientific computing in Python
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from matplotlib.dates import DateFormatter

def togregorian(x):
    gregoriandate = dt.date(1, 1, 1) + dt.timedelta(days=x)
    return gregoriandate

#import data and some preprocessing

Country = 'Uruguay'
Countrytwo = 'Belgium'

url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/locations.csv'
dfcountriescodes = pd.read_csv(url, error_bad_lines=False)
url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations-by-manufacturer.csv'
dfvacbyman = pd.read_csv(url, error_bad_lines=False)
groupdf = dfvacbyman.groupby(['location']).sum()
url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv'
dfvac = pd.read_csv(url, error_bad_lines=False)
groupdf = dfvac[['location','daily_vaccinations']].groupby(['location']).sum()
dfbycountry = dfvac[dfvac['location']==Country]
dfbycountry['date'] = pd.to_datetime(dfbycountry['date'])
dfbycountrylastdate = dfbycountry[dfbycountry['date']==dfbycountry['date'].max()]
dfbycountrytoplot = dfbycountry[['date','people_vaccinated_per_hundred','people_fully_vaccinated_per_hundred']]
        #%% plot per 100 vaccine
fig, ax = plt.subplots()

ax.set_title('Vaccine program in {}'.format(Country))
fig.set_size_inches((12,4))
sns.lineplot(data=dfbycountry, x="date", y="people_vaccinated_per_hundred", legend="full",ax=ax)
sns.lineplot(data=dfbycountry, x="date", y="people_fully_vaccinated_per_hundred",ax=ax)
plt.xlabel('Date')
plt.ylabel('People per 100 of population')
plt.show()

        #%% compare 2 countries
dfbycountrycomp = (dfvac['location']==Country)|(dfvac['location']==Countrytwo)
comparedcountries = dfvac[dfbycountrycomp]

fig, ax = plt.subplots()

ax.set_title('Vaccine program in {} and {}'.format(Country,Countrytwo))
fig.set_size_inches((12,4))
sns.lineplot(data=comparedcountries, x="date", y="people_vaccinated_per_hundred",hue='location',ax=ax)
sns.lineplot(data=comparedcountries, x="date", y="people_fully_vaccinated_per_hundred", hue='location',ax=ax)
plt.xlabel('Date')
plt.ylabel('People per 100 of population')

plt.show()

        #%% vaccine per manufucturer

locations = ['Chile', 'Czechia', 'Germany', 'Iceland', 'Italy', 'Latvia', 'Lithuania', 'Romania', 'United States']
#last day by country of vaccine given
dfvacbyman['date'] = pd.to_datetime(dfvacbyman['date'])
dftrylast = dfvacbyman[dfvacbyman['location']=='Atlantis']
for x in locations:
    dftry = dfvacbyman[dfvacbyman['location']==x]
    dftrylast1 = dftry[dftry['date']==dftry['date'].max()]
    dftrylast=dftrylast.append(dftrylast1)
    #grouping
vacagg = dftrylast.groupby(['location']).sum()
df_group_multiple = dftrylast.groupby(['location', 'vaccine'])
df_group_multiple_es = df_group_multiple['total_vaccinations'].sum()
print(df_group_multiple_es)

        #%% Regression Python
        
dfbycountry['Dateordinal'] = dfbycountry['date'].map(dt.datetime.toordinal)

sns.regplot(x="Dateordinal", y="people_vaccinated_per_hundred", data=dfbycountry).set_title('Manue')
plt.show()   

dfbycountry.replace(np.nan, 0, inplace = True)

X = dfbycountry['Dateordinal'].values.reshape(-1, 1)  # values converts it into a numpy array
Y = dfbycountry['people_vaccinated_per_hundred'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

DateYouWant = dt.date(2021, 10, 22)
Date_In_Ordinal = DateYouWant.toordinal()

X_test = np.array([Date_In_Ordinal]).reshape(1,-1)

regr = linear_model.LinearRegression()
regr.fit(X, Y)
y_pred = regr.predict(X_test)
print("the value will be: {}".format(y_pred[0]))
     
        #%% other way
        

X = dfbycountry['people_vaccinated_per_hundred'].values.reshape(-1, 1)  # values converts it into a numpy array
Y = dfbycountry['Dateordinal'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

X_test = np.array([140]).reshape(1,-1)

regr = linear_model.LinearRegression()
regr.fit(X, Y)
y_pred = regr.predict(X_test)
print("the value will be: {}".format(y_pred[0])) 
        
togregorian(int(y_pred[0]))

str(togregorian(int(y_pred[0])))        
        
          #%% everycountry with a for loop
Country = []
Prediction = []
for x in dfvac['location'].unique():
    
    print(x)
    dfbycountry = dfvac[dfvac['location']==x]
    dfbycountry['date'] = pd.to_datetime(dfbycountry['date'])
    dfbycountry['Dateordinal'] = dfbycountry['date'].map(dt.datetime.toordinal)
    
    dfbycountry = dfbycountry[dfbycountry['people_vaccinated_per_hundred'].notna()]

    
    if dfbycountry['people_vaccinated_per_hundred'].count() > 20:
    
        X = dfbycountry['people_vaccinated_per_hundred'].values.reshape(-1, 1)  # values converts it into a numpy array
        Y = dfbycountry['Dateordinal'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)
    
        X_test = np.array([140]).reshape(1,-1)
    
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
        y_pred = regr.predict(X_test)
        print("the value will be: {}".format(y_pred[0])) 
        togregorian(int(y_pred[0]))
        Country.append(x)
        Prediction.append(int(y_pred[0]))
    
          #%% dataframe of the result and exporting to excel
d = {'Country':Country,'Prediction':Prediction}
A = pd.DataFrame(d)
A['Predictiondate']=A['Prediction'].apply(togregorian)
A['Predictiondate'] = pd.to_datetime(A['Predictiondate'])
B = A.sort_values(by=['Predictiondate'])

B  = B.drop([30,49,71,14,39,14,50,67, 83, 87, 54], axis=0)

L = B.head(20)
            
sns.relplot(x="Country", y="Predictiondate", hue="Country", data=L)

with pd.ExcelWriter('output.xlsx') as writer:  
    B.to_excel(writer, sheet_name='CovidHerd')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


