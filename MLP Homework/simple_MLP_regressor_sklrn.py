# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:49:12 2023

@author: jg
"""


from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X, y = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1)


regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

regr.predict(X_test[:2])

regr.score(X_test, y_test)
