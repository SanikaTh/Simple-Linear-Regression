# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:49:48 2024

@author: HP
"""

'''
A certain food-based company conducted a survey with the help of a fitness 
company to find the relationship between a person’s weight gain and the number
of calories they consumed in order to come up with diet plans for these individuals.
Build a Simple Linear Regression model with calories consumed as the target variable.
Apply necessary transformations and record the RMSE and correlation coefficient values
for different models.''' 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
data=pd.read_csv("D:/Documents/Datasets/calories_consumed.csv")
data.columns
#Index(['Weight gained (grams)', 'Calories Consumed'], dtype='object')

data.head()
'''
 Weight gained (grams)  Calories Consumed
0                    108               1500
1                    200               2300
2                    900               3400
3                    200               2200
4                    300               2500
'''

data.shape
#(14, 2)

data.columns 
# Index(['Weight gained (grams)', 'Calories Consumed'], dtype='object')

plt.xlabel('Weight gained (grams)')
plt.ylabel('Calories Consumed')
plt.scatter(data['Weight gained (grams)'],data['Calories Consumed'],color="red",marker="+")

sns.distplot(data["Weight gained (grams)"])
plt.show()
# Data is right skewed

sns.distplot(data["Calories Consumed"])
plt.show()
# data is lightly right skewed 

data["Calories Consumed"]=np.log(data["Calories Consumed"])

data["Weight gained (grams)"]=np.log(data["Weight gained (grams)"])

sns.distplot(data["Weight gained (grams)"])
plt.show()
# after log transformation, data become normally distributed

sns.distplot(data["Calories Consumed"])
plt.show()
# after log transformation, data become normally distributed

sns.boxplot(data["Weight gained (grams)"])
plt.show()
# no outliers are present

sns.boxplot(data["Calories Consumed"])
plt.show()
# no outliers are present

sns.scatterplot(x=data["Weight gained (grams)"],y=data["Calories Consumed"],color="r")
plt.show()

sns.heatmap(data.corr(),annot=True,fmt="0.02f")
plt.show()
# from this graph, we can see that both columns have strong positive correlation because 
# value of correlation coefficient is 0.92 > 0.8 .

sns.pairplot(data)
plt.show()
               
new_data=data.drop("Calories Consumed", axis="columns")
new_data                 

calories=data["Calories Consumed"]
calories
                 
#create linear regression object
reg=linear_model.LinearRegression()
reg.fit(new_data,calories)
reg.predict([[1100]])

reg.coef_

reg.intercept_                 
              
# y=mx+b(m is coef and b is intercept)
1100* 2.13442296 + 1577.2007020291894
#3925.0659580291895        
                 

# For example, converting 'Weight gained (grams)' to numeric if it's not already numeric
data['Weight gained (grams)'] = pd.to_numeric(data['Weight gained (grams)'], errors='coerce')

# Split the data into features (X) and target variable (y)
X = data[['Weight gained (grams)']]
y = data['Calories Consumed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate correlation coefficient
correlation_coefficient = np.corrcoef(y_test, y_pred)[0, 1]

# Print RMSE and correlation coefficient
print("Root Mean Squared Error (RMSE):", rmse)
print("Correlation Coefficient:", correlation_coefficient)

                 
                 
                 
                 