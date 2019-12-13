# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#%matplotlib inline

digits = load_digits()

digits.data.shape
type(digits.data.shape)
digits.data[:4]

plt.figure(figsize=(20,4))

for index,(image,label) in enumerate(zip(digits.data[0:15],digits.target[0:15])):
    plt.subplot(1,15,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title("Training: %i\n" % label,fontsize=20)

x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.23,random_state=2)

x_train.shape
y_train.shape
x_test.shape
y_test.shape

from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression()
logisticRegression.fit(x_train,y_train)

print(y_test[0])
print(logisticRegression.predict(x_test[0].reshape(1,-1)))

predictions = logisticRegression.predict(x_test)
score = logisticRegression.score(x_test,y_test)
print(score)

cm =metrics.confusion_matrix(y_test,predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt=".3f",linewidths=.5,square=True,cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {}'.format(score)
plt.title(all_sample_title,size=15)

 












