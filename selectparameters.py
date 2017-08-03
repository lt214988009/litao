# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:13:03 2017

@author: litao
"""
import dill
import os
from scipy import linalg
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import svm, metrics,decomposition,linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import time
import pydotplus 
from IPython.display import Image
import math
from operator import itemgetter as get
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import permutation_test_score
import sys
import logging
import pandas as pd
from pandas import Series,DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
######读入数据
data=pd.read_csv("data_regression.csv")
print(data.columns)
#['Height', 'Sex', 'HeightBirth', 'WeightBirth', 'IsEnough', 'Weight',
#       'HeightFather', 'HeightMother', 'MonthsBaby', 'AgeFather', 'AgeMother',
#       'DateFather', 'DateMother', 'Group']
data_reg=data[data['Group']==60]
##挑出特征变量与因变量
data_reg=data_reg.iloc[:,[0,1,2,3,5,6,7,8]]
#'Sex', 'HeightBirth', 'WeightBirth', 'Weight','HeightFather', 'HeightMother', 'MonthsBaby'
#DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
#特征矩阵
x=data_reg.drop('Height',axis=1)
#响应变量/因变量
y=data_reg['Height']
#生成训练集与测试集
x_train,x_test,y_train,y_test=train_test_split(x,y)
print('Train sample number:',len(x_train))
print('Test sample number:',len(x_test))
##标准化特征矩阵
scaler=StandardScaler()
scaler.fit(x_train)
x_train1=scaler.transform(x_train)
x_test1=scaler.transform(x_test)


parameters = {'min_samples_leaf':[4,5,6,7,8,9,10],'min_samples_split':[5,10,15,20,25,30],'max_depth':[2,4,6,8,10],
              'n_estimators':[50,100,150,200]}
randomf=RandomForestRegressor()
randomf=GridSearchCV(randomf,parameters)
randomf.fit(x_train1,y_train)
height4=randomf.predict(x_test1)
print('RandomF TRUE Percent',(abs(height4-y_test)<2).sum()/len(abs(height4-y_test)<2))
print(randomf.best_params_)
randomf=RandomForestRegressor(n_estimators=randomf.best_params_['n_estimators'],max_depth=randomf.best_params_['max_depth'], 
                               random_state=0,min_samples_leaf=randomf.best_params_['min_samples_leaf'],
                              min_samples_split=randomf.best_params_['min_samples_split'])
randomf.fit(x_train1,y_train)
height4=randomf.predict(x_test1)
print('GBRT TRUE Percent',(abs(height4-y_test)<2).sum()/len(abs(height4-y_test)<2))



















parameters = {'loss':['lad', 'huber','quantile'], 'learning_rate':[0.05,0.1, 0.15, 0.2,0.4,0.6,0.8],
              'min_samples_leaf':[4,5,6,7,8,9,10],'min_samples_split':[5,10,15,20,25,30],'max_depth':[2,4,6,8,10],
              'n_estimators':[50,100,150,200]}
gbrt=GradientBoostingRegressor()
gbrt=GridSearchCV(gbrt,parameters)
gbrt.fit(x_train1,y_train)
height4=gbrt.predict(x_test1)
print('GBRT TRUE Percent',(abs(height4-y_test)<2).sum()/len(abs(height4-y_test)<2))
print(gbrt.best_params_)
gbrt=GradientBoostingRegressor(n_estimators=gbrt.best_params_['n_estimators'], learning_rate=gbrt.best_params_['learning_rate'],max_depth=gbrt.best_params_['max_depth'], 
                               random_state=0, loss=gbrt.best_params_['loss'],min_samples_leaf=gbrt.best_params_['min_samples_leaf'],min_samples_split=gbrt.best_params_['min_samples_split'])
gbrt.fit(x_train1,y_train)
height4=gbrt.predict(x_test1)
print('GBRT TRUE Percent',(abs(height4-y_test)<2).sum()/len(abs(height4-y_test)<2))

































