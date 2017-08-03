"""
Created on Fri Jun  2 10:04:17 2017

@author: litao
"""
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import os
import re
import urllib
import time

#读取源文件
data = pd.read_csv('survey131.csv')
#数据范围的筛选
data = data[data['HeightFather'] >130 ]
data = data[data['HeightFather'] <200 ]
data = data[data['HeightMother'] >130 ]
data = data[data['HeightMother'] <190 ]
data = data[data['Height'] >0 ]
data = data[data['Weight'] <190 ]
data = data[data['MonsBaby'] >=0 ]
data = data[(data['DateFather'] > 14) & (data['DateFather'] <= 65) ] #生育年龄
data = data[(data['DateMother'] > 14) & (data['DateMother'] <= 55) ] #生育年龄
data = data[(data['AgeFather'] > 14) & (data['AgeFather'] <= 80) ]#现在的年龄
data = data[(data['AgeMother'] > 14) & (data['AgeMother'] <= 80) ]


#数据分组将父母身高换成区间
group_names = ['A','B','C','D','E','F','G' ]
group_months=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,
              45,46,47,48,49,50,51]
hei_fa = data ['HeightFather']
hei_mo = data ['HeightMother']
bins_father = [hei_fa.min()-1,160,165,170,175,180,185,hei_fa.max()]
bins_mother = [hei_mo.min()-1,155,158,160,162,163,166,hei_mo.max()]

data['HeightFatherGroup'] = pd.cut(data['HeightFather'],bins_father,labels = group_names)
data['HeightMotherGroup'] = pd.cut(data['HeightMother'],bins_mother,labels = group_names)
#获取IP所在省份城市
#data['Ip_local']=data['Ip_1']
#data['Ip_local']=list(map(lookup,data['Ip_1']))
#异常值的筛选
height_25 = data['Height'].quantile(0.25)
height_75 = data['Height'].quantile(0.75)
heightFather_25 = data['HeightFather'].quantile(0.25)
heightFather_75 = data['HeightFather'].quantile(0.75)
heightMother_25 = data['HeightMother'].quantile(0.25)
heightMother_75 = data['HeightMother'].quantile(0.75)
height_up = height_75 + (height_75 + height_25)*1.5
height_down = height_25 - (height_75 + height_25)*1.5
heightFather_up = heightFather_75 + (heightFather_75 + heightFather_25)*1.5
heightFather_down = heightFather_25 - (heightFather_75 + heightFather_25)*1.5
heightMother_up = heightMother_75 + (heightMother_75 + heightMother_25)*1.5
heightMother_down = heightMother_25 - (heightMother_75 + heightMother_25)*1.5
data = data[(data['Height'] <= height_up) & (data['Height'] >= height_down) ]
data = data[(data['HeightFather'] <= heightFather_up) & (data['HeightFather'] >= heightFather_down) ]
data = data[(data['HeightMother'] <= heightMother_up) & (data['HeightMother'] >= heightMother_down) ]

#最终的筛选,每列的取值范围

# 保存清洗干净的数据到本地
isExists=os.path.exists("survey131_final.csv")
if isExists:
    os.remove("survey131_final.csv")
data.to_csv("survey131_final.csv",encoding='utf-8')
