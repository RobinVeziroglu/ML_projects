# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:14:26 2022

@author: a403922
"""

import pandas  as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

path_train="C:\\Users\\A403922\\Github_Demo\\Project_1\\Regression\\train.csv"
path_test="C:\\Users\\A403922\\Github_Demo\\Project_1\\Regression\\test.csv"
df_train=pd.read_csv(path_train)
df_test=pd.read_csv(path_test)

#Quick data-cleaning
#Are there rows with missing values?
#print(df_train.isnull().any(axis = 1).sum())
#We have 1 row with a missing value

df_train=df_train.dropna(how='any',axis=0)


#This is a case of simple linear regression, we want to use the independent variable x
#to predict the response variably y.
#y_i=beta_0+beta_1*x_i

x=np.array([df_train.x.to_list()]).reshape(-1,1)
y=np.array([df_train.y.to_list()]).reshape(-1,1)

model=LinearRegression()

model.fit(x,y)

R2_score=model.score(x, y) # coefficient of determination R^2
print(R2_score)

intercept=model.intercept_
print('Estimated y-intercept for the regression-line is {}'.format(intercept))

slope=model.coef_
print('Estimated slope for the regression line is {}'.format(slope))
