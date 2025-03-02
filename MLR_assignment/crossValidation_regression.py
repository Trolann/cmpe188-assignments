# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:00:16 2021

@author: jg
"""


'''
cross validation 

The following sample code implements the k-fold cross validation 
This sample code shows the application of cross-validation to a regression problem
However, the technique equally applies to all kind of model comparison, i.e. model assessment
regardless of what type of models are under considration. 

'''

from sklearn.linear_model import LinearRegression
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE


# load dataset. You should put the dataset in the current directory or provide a path to the where
# the data is located.
dir_path = '../eda_assignment/'
filename = 'boston.csv'
names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad','tax','ptratio','black','istat','medv']
dataframe = read_csv(dir_path + filename)
data1 = dataframe.drop(dataframe.columns[0], axis=1)
array = data1.values
X = array[:,0:13]
Y = array[:,13]


NUM_FEATURES = 12 # this is kind of arbitrary but you should get an idea by observing the scatter plots and correlation.
model = LinearRegression()
rfe_12 = RFE(model, n_features_to_select = NUM_FEATURES)
fit = rfe_12.fit(X, Y)
print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Ranking:", fit.ranking_)
# calculate the score for the selected features
score = rfe_12.score(X,Y)
print("Model Score with selected features 12 is: ", score)
# now change number of features to 5
NUM_FEATURES = 11
rfe_11 = RFE(model, n_features_to_select = NUM_FEATURES)
fit = rfe_11.fit(X, Y)
print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Ranking:", fit.ranking_)
# calculate the score for the selected features
score = rfe_11.score(X,Y)
print("Model Score with selected features 5 is: ", score)

'''NUM_FEATURES = 10 # this is kind of arbitrary but you should get an idea by observing the scatter plots and correlation.
model = LinearRegression()
rfe_10 = RFE(model, NUM_FEATURES)

# now change number of features to 5
NUM_FEATURES = 9
rfe_9 = RFE(model, NUM_FEATURES)

NUM_FEATURES = 7 # this is kind of arbitrary but you should get an idea by observing the scatter plots and correlation.
model = LinearRegression()
rfe_7 = RFE(model, NUM_FEATURES)

# now change number of features to 5
NUM_FEATURES = 5
rfe_5 = RFE(model, NUM_FEATURES)'''

# prepare models
models = []

#######################################################################################
# Prepare 6 multiple linear regression model by reviewing 
# the multiple linear regression lab (RFE model). Choose a model with 12, 11, 10, 9, 7, and 5 features
########################################################################################
models.append(('rfe12', rfe_12))
models.append(('ref11', rfe_11))

# evaluate each model in turn

results = []
names = []

scoring = 'r2'
for name, model in models:
	kfold = KFold(n_splits=10) 
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: mean:%f std:(%f) var:(%f)" % (name, cv_results.mean(), cv_results.std(), cv_results.var())
	print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


########################################################
# compare the result and comment on your findings.