import pandas as pd
import numpy as np
import os
import shutil
#定义可用性标准
usableStd=10 #方法可用性至少达到10人
usableNum=18 #2/3的方法(27种方法)达到usableStd标准

#数据解析,提取症状\诱因\方法数据
filename = "data0526.csv"
clean_df = pd.read_csv(filename, na_values='NA')
symptom = clean_df.iloc[:,0:18] #症状
incentive = clean_df.iloc[:,18:29] #诱因
treatment = clean_df.iloc[:,32:59] #方法
effect = clean_df.iloc[:,59] #疗效
type0 = clean_df.iloc[:,60] #type0
type1 =  clean_df.iloc[:,61] #type1

#baseline 有效率
better = treatment.loc[effect == 0,].sum(axis=0,skipna=True)
nochange = treatment.loc[effect == 1,].sum(axis=0,skipna=True)
worse = treatment.loc[effect == 2,].sum(axis=0,skipna=True)
total = treatment.sum(axis=0,skipna=True)

baseline_rate = sum(better - worse)/sum(total)
print(baseline_rate)

#type0有效率
type_label = list(set(type0))
n_num_0 = len(type_label)
total_better_0 = 0
total_worse_0 = 0
total_all_0 = 0
for i in range(n_num_0):
    clster_i = treatment.loc[type0 == type_label[i],]
    better = clster_i.loc[effect == 0,].sum(axis=0,skipna=True)
    worse  = clster_i.loc[effect == 2,].sum(axis=0,skipna=True)
    all_0    = clster_i.sum(axis=0,skipna=True)
    all_0[all_0 < usableStd] = float("Inf")
    percent_0 = (better -worse)/all_0
    percent_0.sort_values(inplace=True,ascending=False)
    top_index_0 = percent_0.index[0]
    total_better_0 += better.loc[top_index_0]
    total_worse_0 += worse.loc[top_index_0]
    total_all_0 += all_0.loc[top_index_0]	 
type0_rate = (total_better_0 - total_worse_0)/total_all_0
print(type0_rate)

#type1有效率
type_label = list(set(type1))
n_num_1 = len(type_label)
total_better_1 = 0
total_worse_1 = 0
total_all_1 = 0
for i in range(n_num_1):
    clster_i = treatment.loc[type1 == type_label[i],]
    better = clster_i.loc[effect == 0,].sum(axis=0,skipna=True)
    worse  = clster_i.loc[effect == 2,].sum(axis=0,skipna=True)
    all_1    = clster_i.sum(axis=0,skipna=True)
    all_1[all_1 < usableStd] = float("Inf")
    percent_1 = (better -worse)/all_1
    percent_1.sort_values(inplace=True,ascending=False)
    top_index_1 = percent_1.index[0]
    total_better_1 += better.loc[top_index_1]
    total_worse_1 += worse.loc[top_index_1]
    total_all_1 += all_1.loc[top_index_1]               	 
type1_rate = (total_better_1 - total_worse_1)/total_all_1
print(type1_rate)

#def type1_rate(targets, labels, effect, n_clusters, usableStd=0):
    # 聚类指标计算不需要变动
#    total_better = 0
#    total_worse = 0
#    total_all = 0
#    for i in range(n_clusters):
#        clstr_i = targets.loc[labels == i, ]
#        better = clstr_i.loc[effect == 0, ].sum(axis=0, skipna=True)
#        worse = clstr_i.loc[effect == 2, ].sum(axis=0, skipna=True)
#        total = clstr_i.sum(axis=0, skipna=True)
#        total.loc[total < usableStd] = float("inf")
#
#        percent = (better - worse) / total
#        percent.sort_values(inplace=True, ascending=False)
#       top_idx = percent.index[0]  # Select best method in cluster_i
#
#        total_better += better.loc[top_idx]
#        total_worse += worse.loc[top_idx]
#        total_all += total.loc[top_idx]
#
#    return (total_better - total_worse) / total_all
#print(type1_rate(treatment,type1,effect,n_num_1))	
	
	
    
