# -*- coding: utf-8 -*-
"""
Created on Thu May 27 01:18:01 2021

@author: batuh
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error

#ignore warnings
import warnings  
warnings.filterwarnings('ignore')

data = pd.read_csv("diabetes.csv")

head = data.head()

shape = data.shape

#correlation
#Glucose, BMI, Age, Pregnancies have good correlation with Outcome
"""
corr = data.corr()
plt.figure()
sns.heatmap(corr, annot=True)
plt.show()
"""

null = data.isnull().sum()
na = data.isna().sum()

#zero cant acceptable for this columns
zero_not_accepted = ["Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI"]

print("Number of zeros of Glucose column is "+str(data.Glucose[data.Glucose==0].count()))
print("Number of zeros of BloodPressure column is "+str(data.BloodPressure[data.BloodPressure==0].count()))
print("Number of zeros of SkinThickness column is "+str(data.SkinThickness[data.SkinThickness==0].count()))
print("Number of zeros of Insulin column is "+str(data.Insulin[data.Insulin==0].count()))
print("Number of zeros of BMI column is "+str(data.BMI[data.BMI==0].count()))

zero_not_accepted = ["Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI"]
for i in zero_not_accepted:
    data[i].replace(0, data[i].mean(), inplace=True)
    
#splitting dataset into train%80 and test%20
y = data.Outcome
X = data.drop(["Outcome"], axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                    test_size=0.20, random_state=0)

# APPLYING MACHINE LEARNING ALGORITHMS
machine_learning_algorithms = (SVC(), LogisticRegression(), KNeighborsClassifier(),
                               GaussianNB(), RandomForestClassifier(n_estimators=5))

ml_names = ("SVC", "Logistic Regression", "KNN", "Naive Bayes", "RandomForest")
for ml, ml_name in zip(machine_learning_algorithms, ml_names):
    # We split the data into train(%80) and test(%20)
    clf = ml
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    print("{} Accuracy: %".format(ml_name), 100 - mean_absolute_error(y_test, predict) * 100)