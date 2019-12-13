#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:38:23 2019

@author: bikash
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import math

diabetes_file = '/media/bikash/Work/Study/ML/SL datasets/Machine Learning Full/KNN/diabetes.csv'

df = pd.read_csv(diabetes_file)

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']

for column in zero_not_accepted:
    #print(column)
    df[column] = df[column].replace(0,np.NaN)
    mean = int(df[column].mean(skipna=True))
    #print(mean)
    df[column] = df[column].replace(np.NaN,mean)
    #print(df[column].mean())

X = df.iloc[:,0:8]
y=df.iloc[:,8]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(x_train)
X_test = sc_X.fit_transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
ac_score = accuracy_score(y_test,y_pred)
print('accuracy_score: {}'.format(ac_score))
cm = confusion_matrix(y_test,y_pred)
print(cm)
f1 = f1_score(y_test,y_pred)
print('f1_score: {}'.format(f1))


