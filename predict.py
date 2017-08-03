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
import xgboost as xgb

def genindexlist(innum,tonum):
    i=innum
    if 0<innum and innum<=tonum:
        indexlist=pd.DataFrame(np.zeros([int(math.factorial(tonum)/math.factorial(i)/math.factorial(tonum-i)),i]))
        #        while j in range(math.factorial(tonum)/math.factorial(i)/math.factorial(tonum-i)):
        if innum==tonum:
            indexlist.loc[0,:]=np.arange(tonum)
        #        elif innum==0:
        #            return 5*np.ones([1,innum])
        elif innum==1:
            indexlist.loc[:,0]=np.arange(tonum).reshape(tonum,1)
        else:
            temp1=pd.DataFrame(np.zeros([int(math.factorial(tonum-1)/math.factorial(i)/math.factorial(tonum-1-i)),i]))
            temp2=pd.DataFrame(np.zeros([int(math.factorial(tonum-1)/math.factorial(i-1)/math.factorial(tonum-i)),i]))
            temp3=pd.DataFrame(np.zeros([int(math.factorial(tonum-1)/math.factorial(i-1)/math.factorial(tonum-i)),i-1]))
            temp3.loc[:,:]=genindexlist(innum-1,tonum-1)
            temp1.loc[:,:]=genindexlist(innum,tonum-1)
           #temp3!=5*np.ones([1,innum-1]):
            temp2.loc[:,0:i-2]=pd.DataFrame.as_matrix(temp3)
            temp2.loc[:,i-1]=(tonum-1)*np.ones([len(temp2),1])
            indexlist.loc[0:temp1.shape[0]-1,:]=pd.DataFrame.as_matrix(temp1)
            indexlist.loc[temp1.shape[0]:,:]=pd.DataFrame.as_matrix(temp2) 
        data_index=pd.DataFrame.as_matrix(indexlist)
        return data_index.astype('int64')
    else:
        raise ValueError('Must satisfy 0<a<=b')
        
        

def modelsetting(Xrange,Y,minin,estimatorvec):
    length_var=Xrange.shape[1]
    Score=[]
    for j in range(minin,length_var+1):
        indexlist=genindexlist(j,length_var)
        for i in range(len(indexlist)): # i is total number of independent variables
            xvarlist=list(indexlist[i,:])
            X=Xrange.iloc[:,xvarlist]    
            for k in range(len(estimatorvec)):
                x_train,x_test,y_train,y_test=train_test_split(X,Y)
                scaler=StandardScaler()
                scaler.fit(x_train)
                x_train1=scaler.transform(x_train)
                x_test1=scaler.transform(x_test)
                estimatorvec[k].fit(x_train1,y_train)
                height=estimatorvec[k].predict(x_test1)
                score_2=(abs(height-y_test)<2).sum()/len(abs(height-y_test)<2)
                Score.append([j,xvarlist,k,score_2])
                
    return pd.DataFrame(Score)

        
def reducedim(X):
    pca.fit(X)
    print(pca.explained_variance_)
    pca.n_components =10
    X_red = pca.fit_transform(X)
    return X_red
######模型
pca = decomposition.PCA()
knn = KNeighborsClassifier()
lr = linear_model.LinearRegression()
rla = linear_model.Lasso()
lg = linear_model.LogisticRegression(max_iter=1000, multi_class='multinomial',penalty='l2',solver='newton-cg')
lsvc = svm.LinearSVC()
rsvc = svm.SVC(kernel='rbf')
psvc = svm.SVC(kernel='poly',degree=3)
poknn=Pipeline([('poly', PolynomialFeatures(degree=3)),('knn', KNeighborsClassifier())])
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
nb = GaussianNB()
tree = tree.DecisionTreeClassifier()
ml=MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',\
       beta_1=0.9, beta_2=0.999, early_stopping=False,\
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',\
       learning_rate_init=0.001, max_iter=200, momentum=0.9,\
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,\
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,\
       warm_start=False)
skf=StratifiedKFold(n_splits=10, random_state=0, shuffle=False)

mlp=MLPRegressor(hidden_layer_sizes=(7,7),max_iter=500,solver='lbfgs', random_state=0)

ada=AdaBoostRegressor(n_estimators=100,learning_rate=0.2,loss='square', random_state=0)
'''
AdaBoost算法参数详解
max_features: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；如果是"log2"意味着划分时最多考虑log2Nlog2N个特征；
              如果是"sqrt"或者"auto"意味着划分时最多考虑N−−√N个特征。如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比
max_depth:决策树最大深
min_weight_fraction_leaf:叶子节点最小的样本权重
min_samples_leaf:叶子节点最少样本数
max_leaf_nodes:最大叶子节点数
base_estimator：AdaBoostClassifier和AdaBoostRegressor都有，即我们的弱分类学习器或者弱回归学习器。理论上可以选择任何一个分类或者回归学习器，
                不过需要支持样本权重。我们常用的一般是CART决策树或者神经网络MLP。默认是决策树，即AdaBoostClassifier默认使用CART分类树DecisionTreeClassifier，
                而AdaBoostRegressor默认使用CART回归树DecisionTreeRegressor。另外有一个要注意的点是，如果我们选择的AdaBoostClassifier算法是SAMME.R，
                则我们的弱分类学习器还需要支持概率预测，也就是在scikit-learn中弱分类学习器对应的预测方法除了predict还需要有predict_proba。
algorithm：这个参数只有AdaBoostClassifier有。主要原因是scikit-learn实现了两种Adaboost分类算法，SAMME和SAMME.R。两者的主要区别是弱学习器权重的度量，
           SAMME使用了和我们的原理篇里二元分类Adaboost算法的扩展，即用对样本集分类效果作为弱学习器权重，而SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重   
loss：这个参数只有AdaBoostRegressor有，Adaboost.R2算法需要用到。有线性‘linear’, 平方‘square’和指数 ‘exponential’三种选择, 默认是线性
n_estimators： AdaBoostClassifier和AdaBoostRegressor都有，就是我们的弱学习器的最大迭代次数，或者说最大的弱学习器的个数
learning_rate: 学习率           
'''
randomf=RandomForestRegressor(max_depth=10,n_estimators=100,min_samples_leaf=6, min_samples_split=20,random_state=0)
randomfc=RandomForestClassifier(max_depth=10,n_estimators=100,min_samples_leaf=6, min_samples_split=20,random_state=0)
'''
RandomForest参数详解
n_estimators=10：决策树的个数，越多越好，但是性能就会越差，至少100左右（具体数字忘记从哪里来的了）可以达到可接受的性能和误差率。 
bootstrap=True：是否有放回的采样。  
oob_score=False：oob（out of band，带外）数据，即：在某次决策树训练中没有被bootstrap选中的数据。多单个模型的参数训练，
                 我们知道可以用cross validation（cv）来进行，但是特别消耗时间，而且对于随机森林这种情况也没有大的必要，所以就用这个数据对决策树模型进行验证，算是一个简单的交叉验证。性能消耗小，但是效果不错。  
n_jobs=1：并行job个数。这个在ensemble算法中非常重要，尤其是bagging（而非boosting，因为boosting的每次迭代之间有影响，所以很难进行并行化），因为可以并行从而提高性能。1=不并行；n：n个并行；-1：CPU有多少core，就启动多少job
warm_start=False：热启动，决定是否使用上次调用该类的结果然后增加新的。  
class_weight=None：各个label的权重。  
max_depth: (default=None)设置树的最大深度，默认为None，这样建树时，会使每一个叶节点只有一个类别，或是达到min_samples_split。
min_samples_split:根据属性划分节点时，每个划分最少的样本数。
min_samples_leaf:叶子节点最少的样本数。
max_leaf_nodes: (default=None)叶子树的最大样本数。
min_weight_fraction_leaf: (default=0) 叶子节点所需要的最小权值
verbose:(default=0) 是否显示任务进程
criterion: ”gini” or “entropy”(default=”gini”)是计算属性的gini(基尼不纯度)还是entropy(信息增益)，来选择最合适的节点。
splitter: ”best” or “random”(default=”best”)随机选择属性还是选择不纯度最大的属性，建议用默认。
max_features: 选择最适属性时划分的特征不能超过此值。
'''
svr=svm.SVR(C=1.0,epsilon=0.1,kernel='linear')
'''
SVM参数详解
C：C-SVC的惩罚参数,默认值是1.0C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，
    趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，
    允许容错，将他们当成噪声点，泛化能力较强。
kernel ：核函数，默认是rbf，可以是'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' 
degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
probability ：是否采用概率估计？.默认为False
shrinking ：是否采用shrinking heuristic方法，默认为true
tol ：停止训练的误差值大小，默认为1e-3
class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
verbose ：允许冗余输出
max_iter ：最大迭代次数。-1为无限制。
decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3    
random_state ：数据洗牌时的种子值，int值
'''
#kernel:linear,rbf,sigmoid ,polynomial
gbrt=GradientBoostingRegressor(n_estimators=100, learning_rate=0.2,max_depth=10, 
                               random_state=0, loss='lad',min_samples_leaf=6,min_samples_split=20)
#loss:lad(Least absolute deviation),Huber('huber'),Quantile('quantile')
'''
GBRT参数详解
min_ samples_split 
定义了树中一个节点所需要用来分裂的最少样本数。
可以避免过度拟合(over-fitting)。如果用于分类的样本数太小，模型可能只适用于用来训练的样本的分类，而用较多的样本数则可以避免这个问题。
但是如果设定的值过大，就可能出现欠拟合现象(under-fitting)。因此我们可以用CV值（离散系数）考量调节效果。

min_ samples_leaf 
定义了树中终点节点所需要的最少的样本数。
同样，它也可以用来防止过度拟合。
在不均等分类问题中(imbalanced class problems)，一般这个参数需要被设定为较小的值，因为大部分少数类别（minority class）含有的样本都比较小。

min_ weight_ fraction_leaf 
和上面min_ samples_ leaf很像，不同的是这里需要的是一个比例而不是绝对数值：终点节点所需的样本数占总样本数的比值。
#2和#3只需要定义一个就行了

max_ depth 
定义了树的最大深度。
它也可以控制过度拟合，因为分类树越深就越可能过度拟合。
当然也应该用CV值检验。

max_ leaf_ nodes 
定义了决定树里最多能有多少个终点节点。
这个属性有可能在上面max_ depth里就被定义了。比如深度为n的二叉树就有最多2^n个终点节点。
如果我们定义了max_ leaf_ nodes，GBM就会忽略前面的max_depth。

max_ features 
决定了用于分类的特征数，是人为随机定义的。
根据经验一般选择总特征数的平方根就可以工作得很好了，但还是应该用不同的值尝试，最多可以尝试总特征数的30%-40%.
过多的分类特征可能也会导致过度拟合。

learning_ rate 
这个参数决定着每一个决定树对于最终结果（步骤2.4）的影响。GBM设定了初始的权重值之后，每一次树分类都会更新这个值，而learning_ rate控制着每次更新的幅度。
一般来说这个值不应该设的比较大，因为较小的learning rate使得模型对不同的树更加稳健，就能更好地综合它们的结果。

n_ estimators 
定义了需要使用到的决定树的数量（步骤2）
虽然GBM即使在有较多决定树时仍然能保持稳健，但还是可能发生过度拟合。所以也需要针对learning rate用CV值检验。
subsample
训练每个决定树所用到的子样本占总样本的比例，而对于子样本的选择是随机的。
用稍小于1的值能够使模型更稳健，因为这样减少了方差。
一把来说用~0.8就行了，更好的结果可以用调参获得。
loss
指的是每一次节点分裂所要最小化的损失函数(loss function)
对于分类和回归模型可以有不同的值。一般来说不用更改，用默认值就可以了，除非你对它及它对模型的影响很清楚。

init 
它影响了输出参数的起始化过程
如果我们有一个模型，它的输出结果会用来作为GBM模型的起始估计，这个时候就可以用init

random_ state 
作为每次产生随机数的随机种子
使用随机种子对于调参过程是很重要的，因为如果我们每次都用不同的随机种子，即使参数值没变每次出来的结果也会不同，这样不利于比较不同模型的结果。
任一个随即样本都有可能导致过度拟合，可以用不同的随机样本建模来减少过度拟合的可能，但这样计算上也会昂贵很多，因而我们很少这样用

verbose 
决定建模完成后对输出的打印方式： 
0：不输出任何结果（默认）
1：打印特定区域的树的输出结果
>1：打印所有结果
warm_ start 
这个参数的效果很有趣，有效地使用它可以省很多事
使用它我们就可以用一个建好的模型来训练额外的决定树，能节省大量的时间，对于高阶应用我们应该多多探索这个选项。

presort 
决定是否对数据进行预排序，可以使得树分裂地更快。
默认情况下是自动选择的，当然你可以对其更改
'''
rr=linear_model.Ridge(alpha=1e3)
rrcv=linear_model.RidgeCV(alphas=[0.1,1.0,10])
lr=linear_model.LinearRegression()
lasso=linear_model.Lasso()
lassocv=linear_model.LassoCV(alphas=[0.1,1.0,10])
byslr=linear_model.BayesianRidge()
ard=linear_model.ARDRegression()


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
##神经网络预测
mlp.fit(x_train1,y_train)
height=mlp.predict(x_test1)
print('MLP TRUE Percent',(abs(height-y_test)<2).sum()/len(abs(height-y_test)<2))
#DecisionTreeRegressor(max_depth=20, min_samples_split=2, min_samples_leaf=5),
###Adaboost预测
ada.fit(x_train1,y_train)
height1=ada.predict(x_test1)
print('AdaBoost TRUE Percent',(abs(height1-y_test)<2).sum()/len(abs(height1-y_test)<2))
##随机森林预测
randomf.fit(x_train1,y_train)
height2=randomf.predict(x_test1)
print('RandomForest TRUE Percent',(abs(height2-y_test)<2).sum()/len(abs(height2-y_test)<2))
##SVM预测
svr.fit(x_train1,y_train)
height3=svr.predict(x_test1)
print('SVM TRUE Percent',(abs(height3-y_test)<2).sum()/len(abs(height3-y_test)<2))
##Gradient Boosting Decision Tree预测
gbrt.fit(x_train1,y_train)
height4=gbrt.predict(x_test1)
print('GBRT TRUE Percent',(abs(height4-y_test)<2).sum()/len(abs(height4-y_test)<2))
##分类 预测
randomfc.fit(x_train1,y_train)
height5=randomfc.predict(x_test1)
print('RandomForestClassified TRUE Percent',(abs(height5-y_test)<2).sum()/len(abs(height5-y_test)<2))
#lhu = linear_model.HuberRegressor(max_iter=1000)



###XGBoost预测
xgb_train = xgb.DMatrix(x_train1, label=y_train)
xgb_test = xgb.DMatrix(x_test1,label=y_test)
params={
'booster':'gbtree',
'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
#'nthread':7,# cpu 线程数 默认最大
'eta': 0.007, # 如同学习率
'min_child_weight':3, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'max_depth':6, # 构建树的深度，越大越容易过拟合
'gamma':0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样 
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'alpha':0, # L1 正则项参数
#'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
#'objective': 'multi:softmax', #多分类的问题
#'num_class':10, # 类别数，多分类与 multisoftmax 并用
'seed':1000, #随机种子
#'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 100 # 迭代次数
watchlist = [(xgb_train, 'train'),(xgb_test, 'val')]
#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
height13= model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
print('XGBoost TRUE Percent',sum(abs(height13-y_test)<2)/len(height13))



##########线性预测
data_reg1=data[data['Group']==60]
##挑出特征变量与因变量
data_reg1=data_reg1.iloc[:,[0,1,5,6,7,8]]
#'Sex', 'HeightBirth', 'WeightBirth', 'Weight','HeightFather', 'HeightMother', 'MonthsBaby'
#DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
#特征矩阵
x1=data_reg1.drop('Height',axis=1)
#响应变量/因变量
y1=data_reg1['Height']
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


#####建立相应的函数，用所有的模型进行预测
estimatorvec=[mlp,ada,randomf,svr,gbrt,randomfc,rr,rrcv,lasso,lassocv,lr,byslr] 
combinedmx=modelsetting(x,y,3,estimatorvec)
combinedmx.columns=['变量个数','变量','模型','预测准确率']


##利用sklearn.model_selection中GridSearchCV函数挑选最优参数的
##缺点太明显：速度慢的令人发指
#parameters = {'loss':['lad', 'huber','quantile'], 'learning_rate':[0.05,0.1, 0.15, 0.2,0.4,0.6,0.8],
#              'min_samples_leaf':[4,5,6,7,8,9,10],'min_samples_split':[5,10,15,20,25,30],'max_depth':[2,4,6,8,10],
#              'n_estimators':[50,100,150,200]}
#gbrt=GradientBoostingRegressor()
#gbrt=GridSearchCV(gbrt,parameters)
#gbrt.fit(x_train1,y_train)
#height4=gbrt.predict(x_test1)
#print('GBRT TRUE Percent',(abs(height4-y_test)<2).sum()/len(abs(height4-y_test)<2))
#print(gbrt.best_params_)
#gbrt=GradientBoostingRegressor(n_estimators=gbrt.best_params_['n_estimators'], learning_rate=gbrt.best_params_['learning_rate'],max_depth=gbrt.best_params_['max_depth'], 
#                               random_state=0, loss=gbrt.best_params_['loss'],min_samples_leaf=gbrt.best_params_['min_samples_leaf'],min_samples_split=gbrt.best_params_['min_samples_split'])
#gbrt.fit(x_train1,y_train)
#height4=gbrt.predict(x_test1)
#print('GBRT TRUE Percent',(abs(height4-y_test)<2).sum()/len(abs(height4-y_test)<2))


##将结果输出到本地
isExists=os.path.exists("predict.csv")
if isExists:
    os.remove("predict.csv")
combinedmx.to_csv("predict.csv",encoding='utf-8')