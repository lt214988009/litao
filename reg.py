# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:50:22 2016

@author: ZFS
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

pca = decomposition.PCA()
knn = KNeighborsClassifier()
lr = linear_model.LinearRegression()
#lhu = linear_model.HuberRegressor(max_iter=1000)
rr = linear_model.Ridge(alpha=1e5)
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


def genindexlist(innum,tonum):
    i=innum
    if 0<innum and innum<=tonum:
        indexlist=pd.DataFrame(np.zeros([math.factorial(tonum)/math.factorial(i)/math.factorial(tonum-i),i]))
        #        while j in range(math.factorial(tonum)/math.factorial(i)/math.factorial(tonum-i)):
        if innum==tonum:
            indexlist.loc[0,:]=np.arange(tonum)
        #        elif innum==0:
        #            return 5*np.ones([1,innum])
        elif innum==1:
            indexlist.loc[:,0]=np.arange(tonum).reshape(tonum,1)
        else:
            temp1=pd.DataFrame(np.zeros([math.factorial(tonum-1)/math.factorial(i)/math.factorial(tonum-1-i),i]))
            temp2=pd.DataFrame(np.zeros([math.factorial(tonum-1)/math.factorial(i-1)/math.factorial(tonum-i),i]))
            temp3=pd.DataFrame(np.zeros([math.factorial(tonum-1)/math.factorial(i-1)/math.factorial(tonum-i),i-1]))
            temp3.loc[:,:]=genindexlist(innum-1,tonum-1)
            temp1.loc[:,:]=genindexlist(innum,tonum-1)
           #temp3!=5*np.ones([1,innum-1]):
            temp2.loc[:,0:i-2]=pd.DataFrame.as_matrix(temp3)
            temp2.loc[:,i-1]=(tonum-1)*np.ones([len(temp2),1])
            indexlist.loc[0:temp1.shape[0]-1,:]=pd.DataFrame.as_matrix(temp1)
            indexlist.loc[temp1.shape[0]:,:]=pd.DataFrame.as_matrix(temp2)      
        return pd.DataFrame.as_matrix(indexlist)
    else:
        raise ValueError('Must satisfy 0<a<=b')
        
        

def modelsetting(Xrange,Y,minin,estimatorvec,yname):
    for j in range(minin,Xrange.shape[1]+1):
        indexlist=genindexlist(j,Xrange.shape[1])
        for i in range(len(indexlist)): # i is total number of independent variables
            xvarlist=list(indexlist[i,:])
#            xvarlist=pd.Series(np.arange(Xrange.shape[1])).loc[indexlist[i]]
#            xvarlist=pd.DataFrame.as_matrix(xvarlist)
            print('==================================================')
            print('==================================================')
            print('Model specification %d, number of principle components is %d, y=%s'%(i+1,len(xvarlist),yname))
#            timeS=time.time()
#            X=reducedim(data[xvarlist])
#            Y=data[Ycol]
#            timee=time.time()
#            print('Model specification %d, time used for dimension reduction %d'%(i+1,timee-timeS))
            X=Xrange[:,xvarlist]
#            temp=pd.DataFrame(np.zeros([len(X),X.shape[1]+2]))
#            temp.loc[:,0]=data['user_id']
#            temp.loc[:,1:X.shape[1]]=X
#            temp.loc[:,X.shape[1]+1]=Y
#            temp=temp.dropna(axis=0)
#            X=temp.loc[:,1:X.shape[1]].values
#            Y=temp.loc[:,X.shape[1]+1].values
#            print('**************************************************')
#            print('Model specification %d, Kfold estimation'%(i+1))
#            times=time.time()
#            for name in []:
#                method(name,X,Y)  # K-fold 
#            timee=time.time()
#            print('Model specification %d, Kfold estimation accuracy score %.3f, time used is %.d'%(i+1,kfscore,timee-times))
#            print('**************************************************')
            print('Model specification %d, Stratified Kfold estimation'%(i+1))
            time1=time.time()      
#            for tr_id, te_id in skf.split(X,Y):
#                X_tr_sam,X_te_sam=X[tr_id],X[te_id]
#                Y_tr_sam,Y_te_sam=Y[tr_id],Y[te_id]
#                training(X,Y)   #Training
            for k in range(len(estimatorvec)):
#                skfscore=cross_val_score(estimator, X, Y,scoring="neg_mean_squared_error", cv=10)
                times=time.time()
                skfscore, permutation_scores, pvalue = permutation_test_score(estimatorvec[k], X, Y, scoring="accuracy", cv=skf, n_permutations=100, n_jobs=-1)  # 可换别的scoring
                #Plot permutation_scores vs score
                f=plt.figure()
                plt.hist(permutation_scores,10,label='Permutation scores')
                ylim = plt.ylim()
                plt.title('Estimator is %s, i is %d, j is %d'%(str(estimatorvec[k]),i,j))
#                plt.plot(2 * [skfscore], ylim, '--g', linewidth=3,label='Classification Score'' (pvalue %.3f)'%(pvalue))
                plt.vlines(skfscore,0,1,colors='g',linestyles='--',linewidth=3, label='Classification Score'' (pvalue %.3f)'%(pvalue))
#                plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')
#                plt.ylim(ylim)
                plt.legend()
                plt.xlabel('Score')
                plt.show()
                fn='permuscore_'+str(j)+'_'+str(k)+'_'+str(i)+'.png'
                f.savefig(fn)
                timee=time.time()
#                scoremx[j-minin,k,i]=skfscore
#                pvmx[j-minin,k,i]=pvalue
                combinedmx[j-minin,k,i]=skfscore
                combinedmx[j-minin+len(Xrange)-minnumofx+1,k,i]=pvalue
                print('Model specification %d, method = %s, Stratified Kfold estimation accuracy score %.3f, time used is %.d'%(i+1,str(estimatorvec[k]),skfscore,timee-times))
            
#            print('Model specification %d, Stratified Kfold estimation accuracy score %.3f, time used is %.d'%(i+1,skfscore,timee-times))
            timee=time.time()
            print('Model specification %d, total estimation time=%d'%(i+1,timee-time1))
            print('==================================================')
            print('==================================================')
        del indexlist
        # 选出score最高的方法
        aan=Y+'_cvreg.xlsx'
        aa=pd.DataFrame(combinedmx[j-minin,:,:])
        sheetn='Sheet'+str(j-minin)
        aa.to_excel(aan, sheet_name=sheetn)
        bbn=Y+'_cvpvalue.xlsx'
        bb=pd.DataFrame(combinedmx[j-minin+len(Xrange)-minnumofx+1,:,:])
        sheetn='Sheet'+str(j-minin)
        bb.to_excel(bbn, sheet_name=sheetn)
        return combinedmx

        
def reducedim(X):
    pca.fit(X)
    print(pca.explained_variance_)
    pca.n_components =10
    X_red = pca.fit_transform(X)
    return X_red
    
    
    
###################################
#体质数据分析
################################### 
sv15d=pd.read_pickle('sv15d.pkl')
sv15d.drop(['sv15_5','sv15_30','sv15_100','sv15_570','sv15_580','sv15_590','sv15_610','sv15_620'],inplace=True,axis=1)
#values={'sv15_130':'0','sv15_130':'1','sv15_130':'2','sv15_130':'3','sv15_130':'4'}
#datafill=sv15d[sv15d['sv15_130']!='nan']
#row_mask = sv15d.isin(values).all(1)
#datafill=sv15d.where()
datafill=sv15d.loc[:,:'sv15_140']
datafill1=datafill.dropna(axis=0)
datafill1=pd.merge(datafill1,sv15d,on='user_id',how='inner')
datafill1.drop(['sv15_10_y','sv15_20_y','sv15_130_y','sv15_140_y'],inplace=True,axis=1)
varlist15=list(sv15d.columns.values)
#datafill1.loc[:,'sv15_10_x':]=pd.DataFrame(datafill1.loc[:,'sv15_10_x':]).astype(float)
#for j in list(datafill1.columns.values):
#        temp=datafill1[j].str.split(' ',expand=True)
#        datafill1[j]=temp[0].values
datafill1=datafill1.values
for j in range(1,datafill1.shape[1]):
        for k in range(len(datafill1)): 
            if type(datafill1[k,j])==str:
                datafill1[k,j]==np.nan
                datafill1[k,j]=float(datafill1[k,j])
            else:            
                datafill1[k,j]=float(datafill1[k,j])
datafill1=pd.DataFrame(datafill1[:,1:]).astype(float)
for i in range(datafill1.shape[1]):
    datafill1.loc[:,i]=pd.DataFrame(datafill1.loc[:,i]).fillna(value=datafill1.loc[:,i].mean(),axis=1)
varlist15.remove('user_id')
#varlist15=varlist15[4:48]
Xr=pd.DataFrame(np.zeros([44,43]))

varlist15c=list(np.arange(len(varlist15)))
for i in range(len(Xr)):
    varlist15c.remove(varlist15c[i])
    for k in range(len(varlist15c)):
        Xr.loc[i,k]=varlist15c[k]
    varlist15c=list(np.arange(len(varlist15)))
Xr.index=varlist15
MinX=8*np.ones([44,1])
Maxid=np.ones([len(varlist15),3])

for i in range(2,len(varlist15)):
    minnumofx=int(MinX[i])
    Y=pd.DataFrame.as_matrix(datafill1.loc[:,i])
    j=varlist15[i]
    Xrange=reducedim(datafill1.loc[:,list(Xr.loc[j,:])])  
    
#    scoremx=np.zeros(([len(Xrange)-minnumofx+1,8,100]))
#    pvmx=np.zeros(([len(Xrange)-minnumofx+1,8,100]))
#    combinedmx=np.zeros(([2*(len(Xrange)-minnumofx+1),8,100]))
    estimatorvec=[knn,lr,tree,ml,lda]    #poknn,
    combinedmx=modelsetting(Xrange,Y,minnumofx,estimatorvec,j)
    # 挑出score最高的模型
    maxscore=combinedmx.max()
    Maxid[i,:]=np.where(combinedmx==maxscore)


















