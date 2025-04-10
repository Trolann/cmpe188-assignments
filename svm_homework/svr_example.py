# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 07:57:38 2018

@author: jahan
"""

# SVM Regression
from pandas import read_csv
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

dataset = read_csv('C:/Users/jahan/Documents/SaintMaryCollege/Courses/OPS808/Summer 2018/PythonCodeExamples/Baseball_salary.csv')  

dataset.shape
sumNullRws = dataset.isnull().sum()
# remove null elements in data
dataset = dataset.dropna()
# check to see if there is any nulls left
dataset.isnull().sum()

dataset.shape

array = np.log(dataset['Salary'].values)

dataset.loc[:,'SalaryLog'] = pd.Series(array, index=dataset.index)

dataset.describe()  

dataset = dataset.dropna()

dataset.head(20)

X = dataset.loc[:,['Hits', 'Years', 'RBI','Walks', 'Runs', 'PutOuts']]
y = dataset['SalaryLog']  





models = []
results = []
names = []

models.append(('Linear SVR', LinearSVR(epsilon=1.5, random_state = 42)))
models.append(('Kernel SVR', SVR(kernel="rbf", degree = 3, C = 0.2)))
scoring = 'neg_mean_squared_error'


for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f'{name}:  {cv_results.mean():.4}  {cv_results.std():.4}')
    
## boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()