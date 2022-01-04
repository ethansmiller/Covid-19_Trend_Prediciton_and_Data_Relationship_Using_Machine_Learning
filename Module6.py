# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:52:34 2021

@author: Ethan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


#Code for prediction model leveraged from: https://www.youtube.com/watch?v=QIUxPv5PJOY
#Assistance from classmates: Matthew, Michael


# 1 - Trend Prediction

#Dataset acquired from Kaggle: https://www.kaggle.com/antgoldbloom/covid19-data-from-john-hopkins-university

#First we read the csv for global confirmed cases and make sure to select only 3 months of data
df_3_months = pd.read_csv("CONVENIENT_global_confirmed_cases.csv",skiprows=range(1,120),nrows=92)

#Selecting only the columns for date and US case values
df_3_months = df_3_months[['Date', 'US']]
print('\n')
print(df_3_months)

#Plotting our first 3 months of data from 5/20/2020 to 8/20/2020
plt.plot(df_3_months['US'])
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
plt.title('Daily Confirmed Covid-19 Cases in the US (May to August)')
plt.xticks(rotation=70)
plt.show()

#Now we can begin by reading the same csv but now selecting 6 months of data
df = pd.read_csv("CONVENIENT_global_confirmed_cases.csv",skiprows=range(1,120),nrows=184)

#creating a dataframe with only the date and US data
df2 = df[['Date', 'US']]
print('\n')
print(df2)

#Creating a dataframe with only US data
data = df2.filter(['US'])

#converting the dataframe to numpy array
dataset = data.values

#Establishing size of training data set which will be the first 3 months previously plotted
#or half (.5) of the total 6 month data
training_data_len = math.ceil(len(dataset)*.5)

#Scaling training set data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Creating training data set
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(90, len(train_data)):
    x_train.append(train_data[i-90:i,0])
    y_train.append(train_data[i,0])

#Converting training data sets to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

#Reshaping training set 
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Creating LSTM model
model=Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compiling model
model.compile(optimizer='adam', loss='mean_squared_error')

#Training the model with our training sets and allowing for single iteration
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Creating array for the remaining 3 months of values
test_data = scaled_data[training_data_len-90:, :]

#Creating test data sets
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(90, len(test_data)):
    x_test.append(test_data[i-90: i,0])

#Converting to numpy arrays    
x_test = np.array(x_test)

#Reshaping test data sets
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

#Acquiring predicted values based on test data set
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Plotting trained, validated, and finally predicted data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions']=predictions
plt.title('Daily Confirmed Covid-19 Cases in the US (May to Nov)')
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
plt.plot(train['US'])
plt.plot(valid[['US','Predictions']])
plt.legend(['Original', 'Reality', 'Prediction'])
plt.show()
    

# 3 - Data Relationship

#Dataset acquired from Data.gov: https://catalog.data.gov/dataset/md-covid-19-cases-by-county
#First we read the csv for positive Covid-19 cases in Maryland counties
df3 = pd.read_csv("MD_COVID-19_-_Cases_by_County.csv")
#We then modify our dataframe to show only the columns for date and 3 of our selected counties
df3 = df3[['DATE', 'Baltimore', 'Montgomery', 'Kent']]

print('\n\nNumber of Cases in Maryland Counties:\n')
print(df3)

#Plotting relationship between the three county positive case numbers
sns.pairplot(df3, kind="scatter")
plt.show

#Pearson Correlation:
print('\n\nPearson Correlation:\n')
print(df3.corr())

#Chi-Square Test for Independence:

print('\n\nChi-Square Test of Independence between each County:')    

print('\n\nBaltimore cases v. Montgomery cases:\n')
print(chi2_contingency(pd.crosstab(df3.Baltimore, df3.Montgomery)))

print('\n\nBaltimore cases v. Kent cases:\n')
print(chi2_contingency(pd.crosstab(df3.Baltimore, df3.Kent)))

print('\n\nMontgomery cases v. Kent cases:\n')
print(chi2_contingency(pd.crosstab(df3.Montgomery, df3.Kent)))
