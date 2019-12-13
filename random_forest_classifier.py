#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:21:34 2019

@author: bikash
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(0)

iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)

#By manual split

'''
df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)

df['is_train'] = np.random.uniform(0,1,len(df)) <= .75

train,test = df[df['is_train'] == True],df[df['is_train'] == False]
features = df.columns[0:4]
y = pd.factorize(train['species'])[0]
y

rf_clf = RandomForestClassifier(n_jobs=2,random_state=0)
rf_clf.fit(train[features],y)

y_test = pd.factorize(test['species'])[0]
y_pred = rf_clf.predict(test[features])
print(accuracy_score(y_test,y_pred))
'''

X = df
y = pd.Categorical.from_codes(iris.target,iris.target_names).to_list()
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

rf_clf = RandomForestClassifier(n_jobs=2,random_state=0)
rf_clf.fit(x_train,y_train)

y_pred = rf_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))


y_pred = rf_clf.predict(x_train)
print(accuracy_score(y_train,y_pred))
