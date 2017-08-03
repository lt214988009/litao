# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:57:27 2017

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
import time
#读取源文件
frame = pd.read_csv("131.csv",sep=',',skip_blank_lines=True)
idx=frame.index+1
#去除重复值
#frame = frame.drop_duplicates()
#获取相关的字段
frame = frame.loc[:,['user_id', 'qid20', 'qid30', 'qid40', 'qid42', 'qid44', 'qid46','qid48', 'qid60', 'qid70', 'qid80', 'qid82', 'qid90', 'qid100',
                      'qid105', 'qid140', 'qid130', 'qid160', 'qid180', 'qid190', 'qid200','qid210', 'qid220', 'qid230', 'qid50', 'qidreport_ts', 'submit_time','ip']]
#有父母受教育程度的
#frame = frame.loc[:,['user_id', 'qid20', 'qid30', 'qid40', 'qid42', 'qid44', 'qid46','qid48', 'qid60', 'qid70', 'qid80', 'qid82', 'qid90', 'qid100',
#                      'qid105', 'qid140', 'qid130', 'qid160', 'qid180', 'qid190', 'qid200','qid210', 'qid220', 'qid230', 'qid50', 'qidreport_ts', 'submit_time','ip']]

#将变量重命名
frame.columns = ['uID','IsBorned','Sex','BirthDate','IsKnowHeightandWeight','HeightBirth','WeightBirth',
                 'IsEnough','Weight','BirthDateFather','HeightFather','EducationFather','BirthDateMother',
                 'HeightMother','EducationMother','FeedBeforeSix','Feed','IsMedicine','Food','IsBreakfast','IsEgg',
                 'Milk','Exercise','Sleep','Height','WeeksBaby','SubmmitTime','Ip']
frame['Index']=idx
print(len(frame[ (frame['IsKnowHeightandWeight'] == 0)]))
print(len(frame[ (frame['IsKnowHeightandWeight'] != 0)]))
#空值的预处理
frame = frame[ (frame['IsBorned'] != 1)]
print(frame.isnull().astype('int32').sum(axis=0,skipna=True))
frame = frame.fillna('NA')
#针对Isborned列处理，'IsKnowHeightandWeight'处理
frame['IsBorned'] = frame['IsBorned'].replace('NA',1)
#frame['IsKnowHeightandWeight'] = frame['IsKnowHeightandWeight'].replace('NA',1)
frame[frame['uID'].str.contains('test')]='NA'
frame[frame['uID'].str.contains('productdemo')]='NA'
frame[frame['uID'].str.contains('null')]='NA'
frame[frame['uID'].str.contains('</')]='NA'
#针对HeightFather,HeightMother,用众数填充
#frame["fatherHeight"] = frame["fatherHeight"].replace('', mode(frame["fatherHeight"]).mode[0])
#frame["motherHeight"] = frame["motherHeight"].replace('', mode(frame["motherHeight"]).mode[0])
#######初步选取数据

##去掉空值
frame = frame[frame['uID'] != 'NA']
frame = frame[(frame['HeightFather'] != 'NA')&(frame['HeightMother'] != 'NA')]
frame = frame[(frame['Height'] != 'NA')]
frame = frame[(frame['BirthDate'] != 'NA')&(frame['BirthDateMother'] != 'NA')]
#frame = frame[(frame['WeightBirth'] != 'NA')&(frame['IsEnough'] != 'NA')]
frame = frame[(frame['Weight'] != 'NA')&(frame['BirthDateFather'] != 'NA')]
##去掉非数字
frame = frame[(frame['BirthDate'] != 'NaN')]
frame = frame[(frame['BirthDateFather'] != 'NaN') & (frame['BirthDateMother'] != 'NaN') ] 
frame = frame[(frame['HeightBirth'] != 'NaN') & (frame['WeightBirth'] != 'NaN') ] 
frame = frame[(frame['HeightFather'] != 'NaN') & (frame['HeightMother'] != 'NaN') ] 
frame = frame[(frame['Height'] != 'NaN') & (frame['Weight'] != 'NaN') ] 
##筛选出已经出生的
frame = frame[(frame['IsBorned'] == 0)] 
#对特定的列进行清洗
#数据类型的转换
frame['SubmmitTime'] = frame['SubmmitTime'].astype("float64")
frame['BirthDate'] = frame['BirthDate'].astype("float64")
frame['BirthDateFather'] = frame['BirthDateFather'].astype("float64")
frame['BirthDateMother'] = frame['BirthDateMother'].astype("float64")
frame = frame[frame['BirthDateMother'] > 0 ]
frame = frame[frame['BirthDateFather'] > 0 ]
frame = frame[frame['SubmmitTime'] > 0 ]
frame = frame[frame['BirthDate'] > 0 ]
#frame['SubmmitTime'] = pd.to_datetime(pd.DataFrame(frame['SubmmitTime']),unit='ns',origin=pd.Timestamp('1970-01-01 00:00:00'),errors='ignore')
#获取宝贝的月龄
frame['WeeksBaby'] = (frame['SubmmitTime']-frame['BirthDate'])/(1000*3600*24*7*4.28)
frame['WeeksBaby'] = frame['WeeksBaby'].astype("int") 
frame = frame[frame['WeeksBaby'] >= 0 ]
frame['MonthsBaby'] = frame['WeeksBaby']
#爸妈的年龄
frame['AgeFather'] = (frame['SubmmitTime']-frame['BirthDateFather'])/(1000*3600*24*365)
frame['AgeFather'] = frame['AgeFather'].astype("int")
frame = frame[frame['AgeFather']>=0]
frame['AgeMother'] = (frame['SubmmitTime']-frame['BirthDateMother'])/(1000*3600*24*365)
frame['AgeMother'] = frame['AgeMother'].astype("int")
frame = frame[frame['AgeMother']>=0]
#爸妈的生育年龄
frame['DateFather'] = (-frame['BirthDateFather']+frame['BirthDate'])/(1000*3600*24*365)
frame['DateFather'] = frame['DateFather'].astype("int") 
frame= frame[frame['DateFather'] >=0]
frame['DateMother'] = (-frame['BirthDateMother']+frame['BirthDate'])/(1000*3600*24*365)
frame['DateMother'] = frame['DateMother'].astype("int")
frame= frame[frame['DateMother'] >=0]  
##将用毫秒数表示的时间化成日期
frame['SubmmitDate'] = frame['SubmmitTime']/1000
frame['SubmmitDate'] =frame['SubmmitDate'].apply(time.ctime)
frame['BirthDate1'] = frame['BirthDate']/1000
frame['BirthDate1'] =frame['BirthDate1'].apply(time.ctime)
frame['BirthDateMother1'] = frame['BirthDateMother']/1000
frame['BirthDateMother1'] =frame['BirthDateMother1'].apply(time.ctime)
frame['BirthDateFather1'] = frame['BirthDateFather']/1000
frame['BirthDateFather1'] =frame['BirthDateFather1'].apply(time.ctime)
#将IP列进行分列
IP_split=pd.DataFrame((x.split('|') for x in frame['Ip']),index=frame.index,columns=['Ip_1','Ip_2','Ip_3'])
##IP_split=pd.DataFrame((x.split('|') for x in frame['Ip']),index=frame.index,columns=['Ip_1','Ip_2','Ip_3'])
frame= pd.merge(frame,pd.DataFrame(IP_split['Ip_1']),right_index=True,left_index=True)
frame['Ip_1'] = frame['Ip_1'].astype("str") 
##去掉有单位的
#frame['Weight'] = frame['Weight'].str.replace('KG',"")
#frame['Weight'] = frame['Weight'].str.replace('km',"")
#frame['WeightBirth'] = frame['WeightBirth'].str.replace('KG',"")
#frame['WeightBirth'] = frame['WeightBirth'].str.replace('km',"")
#frame['HeightBirth'] = frame['HeightBirth'].str.replace('cm',"")
#frame['HeightBirth'] = frame['HeightBirth'].str.replace('CM',"")
#frame['Height'] = frame['Height'].str.replace('cm',"")
#frame['Height'] = frame['Height'].str.replace('CM',"")
#frame['HeightFather'] = frame['HeightFather'].str.replace('cm',"")
#frame['HeightFather'] = frame['HeightFather'].str.replace('CM',"")
#frame['HeightMother'] = frame['HeightMother'].str.replace('cm',"")
#frame['HeightMother'] = frame['HeightMother'].str.replace('CM',"")

#删除一些无关的列
#del frame['uID']
del frame['Ip']  
frame = frame.drop_duplicates()
#再次进行数据清洗,把一些列中含有单位的去掉，比如50cm
#clo_inx=[4,5,7,9,11,12]
#data.sort_values(by=['HeightBirth','WeightBirth','Weight','HeightFather','HeightMother','Height'],ascending=[0,0,0,0,0,0],inplace=True)
#len_data = len(data)
#for j in clo_inx:
#    for i in range(len_data):
#        string=str(data.iloc[i,j])
#        data.iloc[i,j]=re.findall(r'\d+\.?\d*',string)

# 保存清洗干净的数据到本地
isExists=os.path.exists("survey131_0711.csv")
if isExists:
    os.remove("survey131_0711.csv")
frame.to_csv("survey131_0711.csv",encoding='utf-8')


    
