# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 08:00:05 2018
Updated 4/6/23 for compatibility with new version of SKlearn svm

@author: jahan
"""

import warnings
warnings.filterwarnings("ignore")

# Compare Algorithms
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


# load dataset
filePath = 'C:/Users/jg/Documents/MLData/'
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filePath+filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# we need to standardize the data as the optimization algorithm to find the parameters will run into numerical issues 
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)


# use cross-validation. Although we are building a single classification model
# prepare models
models = []
models.append(('Linear SVC', LinearSVC(C=100, loss = 'hinge', random_state=1, max_iter=1000000)))
models.append(('Kernel SVC', SVC(kernel = 'rbf', degree = 2, C=100.0, random_state=1, max_iter = 1000000)))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle = True)
    cv_results = cross_val_score(model, rescaledX, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
## boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

