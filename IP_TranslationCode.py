# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:32:25 2017

@author: dell
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
import xgboost as xgb
rr=linear_model.Ridge(alpha=1e3)
rrcv=linear_model.RidgeCV(alphas=[0.1,1.0,10])
lr=linear_model.LinearRegression()
lasso=linear_model.Lasso()
lassocv=linear_model.LassoCV(alphas=[0.1,1.0,10])
byslr=linear_model.BayesianRidge()
ard=linear_model.ARDRegression()

##########线性预测
data=pd.read_csv("data_regression.csv")
print(data.columns)
#['Height', 'Sex', 'HeightBirth', 'WeightBirth', 'IsEnough', 'Weight',
#       'HeightFather', 'HeightMother', 'MonthsBaby', 'AgeFather', 'AgeMother',
#       'DateFather', 'DateMother', 'Group']

data_reg=data[data['Group']==60]
##挑出特征变量与因变量
data_reg=data_reg.iloc[:,[0,1,5,6,7,8]]
#'Sex', 'HeightBirth', 'WeightBirth', 'Weight','HeightFather', 'HeightMother', 'MonthsBaby'
#DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
#特征矩阵
x1=data_reg.drop('Height',axis=1)
#响应变量/因变量
y1=data_reg['Height']
#生成训练集与测试集
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1)
##标准化数据
scaler=StandardScaler()
scaler.fit(x1_train)
x1_train2=scaler.transform(x1_train)
x1_test2=scaler.transform(x1_test)

#岭回归
rr.fit(x1_train2,y1_train)
height6=rr.predict(x1_test2)
print('RidgeRegression TRUE Percent',(abs(height6-y1_test)<2).sum()/len(abs(height6-y1_test)<2),rr.coef_,rr.intercept_)
rrcv.fit(x1_train2,y1_train)
height7=rrcv.predict(x1_test2)
print('RidgeCVRegression TRUE Percent',(abs(height7-y1_test)<2).sum()/len(abs(height7-y1_test)<2),rrcv.coef_,rrcv.intercept_)
##Lasso 回归
lasso.fit(x1_train2,y1_train)
height8=lasso.predict(x1_test2)
print('LassoRegression TRUE Percent',(abs(height8-y1_test)<2).sum()/len(abs(height8-y1_test)<2),lasso.coef_,lasso.intercept_)
lassocv.fit(x1_train2,y1_train)
height9=lassocv.predict(x1_test2)
print('LassoCVRegression TRUE Percent',(abs(height9-y1_test)<2).sum()/len(abs(height9-y1_test)<2),lassocv.coef_,lassocv.intercept_)
##一般线性回归
lr.fit(x1_train2,y1_train)
height10=lr.predict(x1_test2)
print('LinearRegression TRUE Percent',(abs(height10-y1_test)<2).sum()/len(abs(height10-y1_test)<2),lr.coef_,lr.intercept_)
##Bayesian 回归
byslr.fit(x1_train2,y1_train)
height11=byslr.predict(x1_test2)
print('BayesianRidgeRegression TRUE Percent',(abs(height11-y1_test)<2).sum()/len(abs(height11-y1_test)<2),byslr.coef_,byslr.intercept_)
#ARD回归
ard.fit(x1_train2,y1_train)
height12=ard.predict(x1_test2)
print('ARDRegression TRUE Percent',(abs(height12-y1_test)<2).sum()/len(abs(height12-y1_test)<2),ard.coef_,ard.intercept_)