# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:36:53 2024

@author: HP
"""

'''
A certain university wants to understand the relationship between students’ 
SAT scores and their GPA. Build a Simple Linear Regression model with GPA as
the target variable and record the RMSE and correlation coefficient values for different models.

Business Objectives - Data Collection: Collect data on SAT scores and corresponding GPAs for a sample of students.

Data Preprocessing: This involves cleaning the data, handling missing values, and splitting the data into training and testing sets.

Model Training: Train a Simple Linear Regression model using the training data.

Model Evaluation: Evaluate the model using metrics like Root Mean Squared Error (RMSE) and correlation coefficient
'''

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data =pd.read_csv("D:/Documents/Datasets/SAT_GPA.csv")
data
data.head()
data.shape
data.columns

plt.xlabel('SAT_scores')
plt.ylabel('GPA')
plt.scatter(data['SAT_Scores'],data['GPA'],color="red",marker="+")
                 
new_data=data.drop("GPA", axis="columns")
new_data                 

sort_t=data["GPA"]
sort_t
                 
#create linear regression object
reg=linear_model.LinearRegression()
reg.fit(new_data,sort_t)

reg.predict([[404]])

reg.coef_

reg.intercept_                 
              
# y=mx+b(m is coef and b is intercept)
404 * 0.00090813 + 2.402871831025617
    

# Split the data into training and testing sets
X = data[['SAT_Scores']]
y = data['GPA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE and correlation coefficient
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
correlation_coefficient = np.corrcoef(y_test, y_pred)[0, 1]

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Correlation Coefficient: {correlation_coefficient}")
