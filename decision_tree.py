#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 07:54:16 2019

@author: bikash
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

loan_repayment_file = '/media/bikash/Work/Study/ML/SL datasets/Machine Learning Full/Decision Tree/Decision_Tree_ Dataset.csv'
wine_quality_file = '/media/bikash/Work/Study/ML/SL datasets/Machine Learning Full/Decision Tree/Decision Tree in R/winequality-red.csv'
wine_quality_file_excel = '/media/bikash/Work/Study/ML/SL datasets/Machine Learning Full/Decision Tree/Decision Tree in R/winequality-red.xls'

#balance_data = pd.read_csv(loan_repayment_file)
#balance_data = pd.read_csv(wine_quality_file)
balance_data = pd.read_excel(wine_quality_file_excel)

X = balance_data.values[:,0:11]
y = balance_data.values[:,11]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)


from sklearn.tree import DecisionTreeClassifier
clf_entropy = DecisionTreeClassifier(criterion='entropy',random_state=100)
clf_entropy.fit(x_train,y_train)


y_pred_dt = clf_entropy.predict(x_test)
print(accuracy_score(y_test,y_pred_dt))

#Preformence comparision with logistic regression
'''
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression(random_state=100)
logisticRegression.fit(x_train,y_train)

y_pred_lr = logisticRegression.predict(x_test)
print(accuracy_score(y_test,y_pred_lr))

'''

#Preformence comparision with Linear regression
'''
X = balance_data.drop(columns=['quality'])
y = balance_data.quality.values
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)

from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(x_train,y_train)

y_pred_linr = linearRegression.predict(x_test)
#print(accuracy_score(y_test,y_pred_linr))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_linr))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_linr))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_linr)))

'''