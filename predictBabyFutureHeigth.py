# -*- coding: UTF-8 -*-
from numpy.linalg import norm
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import random

def main():
    # 读入源数据csv文件
    frame = pd.read_csv('data/survey131clean.csv')
    frame["babyAge"] = (frame["babyAge"]) / 4.28
    frame = frame[frame["babyAge"] > 0]
    frame["babyAge"] = frame["babyAge"].astype("int")
    frame.dropna(axis=0, how='any', inplace=True)
    frame = frame[frame["fatherHeight"] > 100]
    frame = frame[frame["fatherHeight"] < 300]
    frame = frame[frame["motherHeight"] > 100]
    frame = frame[frame["motherHeight"] < 300]

    # 将父母身高转换成区间
    group_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    fa_height = frame["fatherHeight"]
    mo_height = frame["motherHeight"]
    print frame["fatherHeight"].min()
    print frame["fatherHeight"].max()
    print frame["motherHeight"].min()
    print frame["motherHeight"].max()
    bins_father = [fa_height.min(), 160, 165, 170, 175, 180, 185, fa_height.max()]
    #print bins_father
    bins_mother = [mo_height.min(), 145, 150, 155, 160, 165, 170, mo_height.max()]
    #print bins_mother
    fa_height_data = pd.cut(fa_height, bins_father, labels=group_names)
    #print fa_height_data
    mo_height_data = pd.cut(mo_height, bins_mother, labels=group_names)
    #print mo_height_data
    frame.insert(1, 'faHeight', fa_height_data)
    frame.insert(1, 'moHeight', mo_height_data)
    #print frame  # 737088 rows x 6 columns

    # 合并数据集
    num_data = pd.read_csv('data/babyAreaData.csv')
    merge_data = pd.merge(frame, num_data, on=["faHeight", "moHeight", "babyAge"], how="left")
    print merge_data   # 737088 rows x 13 columns

    # 生成1-10的随机数
    rand_data = [random.randrange(1, 11) for i in range(merge_data.shape[0])]
    print rand_data
    rand_data = Series(rand_data)
    merge_data.insert(0, 'age_diff', rand_data)

    '''
    # 由于宝宝270个月以后数据不足，所以对270个月以后数据的age_diff置0
    litter_data = merge_data[merge_data["babyAge"] <= 270]
    #print litter_data
    large_age = merge_data[merge_data["babyAge"] > 270]
    large_age.loc[:, "age_diff"] = 0
    #print large_age
    merge_data = pd.concat([litter_data, large_age], axis=0)
    #print merge_data
    merge_data = merge_data.sort_index()    # 按索引排序
    #print merge_data
    #merge_data.to_csv('data/mergeData.csv', index=False)
    '''

    # 创建一个空的DataFrame，用来存放数据
    # age_diff,fatherHeight,moHeight,faHeight,motherHeight,babyHeight,babyAge,num_5,num_10,num_25,num_50,num_75,num_90,num_95
    res = pd.DataFrame(
        columns=['age_diff', 'fatherHeight', 'moHeight', 'faHeight', 'motherHeight', 'babyHeight', 'babyAge', 'num_5',
                 'num_10', 'num_25', 'num_50', 'num_75', 'num_90', 'num_95'])
    print res

    # 按区间和宝宝年龄排序
    for fa_area in list("A""B""C""D""E""F""G"):
        for mo_area in list("A""B""C""D""E""F""G"):
            sort_data = merge_data[(merge_data["faHeight"] == fa_area) & (merge_data["moHeight"] == mo_area)]
            #print sort_data
            sort_data = sort_data.sort_values(by=["babyAge"])
            #print sort_data
            # 循环添加数据到res中
            res = res.append(sort_data)
            # print res

    # 保存排序完的数据到本地
    res = res[['faHeight', 'moHeight', 'babyAge', 'age_diff', 'fatherHeight',  'motherHeight', 'babyHeight', 'num_5',
               'num_10', 'num_25', 'num_50', 'num_75', 'num_90', 'num_95']]
    res.to_csv('data/sortData.csv', index=False)

    # 算法模块
    res = pd.read_csv('data/sortData.csv')
    print res
    height_list = []
    for i in range(res.shape[0]):
        age = res.ix[i, "babyAge"]
        height = res.ix[i, "babyHeight"]
        num_5 = res.ix[i, "num_5"]
        num_10 = res.ix[i, "num_10"]
        num_25 = res.ix[i, "num_25"]
        num_50 = res.ix[i, "num_50"]
        num_75 = res.ix[i, "num_75"]
        num_90 = res.ix[i, "num_90"]
        num_95 = res.ix[i, "num_95"]
        fa_area = res.ix[i, "faHeight"]
        mo_area = res.ix[i, "moHeight"]
        future_age = res.ix[i, "age_diff"] + age
        cut_data = res[(res["faHeight"] == fa_area) & (res["moHeight"] == mo_area)]

        # 查找未来年龄下的索引
        area_index = cut_data[cut_data["babyAge"] == future_age].index

        # 当小孩年龄大于18岁（12*18=216）时，正常情况下身高不会增长
        if age > 216:
            predict_height = height
        else:
            # 判断area_index是否为空
            if any(area_index):
                area_index = area_index[0]
                # 计算未来年龄下的分位数
                future_num_5 = cut_data.ix[area_index, "num_5"]
                future_num_10 = cut_data.ix[area_index, "num_10"]
                future_num_25 = cut_data.ix[area_index, "num_25"]
                future_num_50 = cut_data.ix[area_index, "num_50"]
                future_num_75 = cut_data.ix[area_index, "num_75"]
                future_num_90 = cut_data.ix[area_index, "num_90"]
                future_num_95 = cut_data.ix[area_index, "num_95"]

                if height < num_5:
                    alpha = 0
                    predict_height = future_num_5 + alpha * (future_num_10 - future_num_5)
                elif height < num_10:
                    alpha = (height - num_5) / (num_10 - num_5)
                    predict_height = future_num_5 + alpha * (future_num_10 - future_num_5)
                elif height < num_25:
                    alpha = (height - num_10) / (num_25 - num_10)
                    predict_height = future_num_10 + alpha * (future_num_25 - future_num_10)
                elif height < num_50:
                    alpha = (height - num_25) / (num_50 - num_25)
                    predict_height = future_num_25 + alpha * (future_num_50 - future_num_25)
                elif height < num_75:
                    alpha = (height - num_50) / (num_75 - num_50)
                    predict_height = future_num_50 + alpha * (future_num_75 - future_num_50)
                elif height < num_90:
                    alpha = (height - num_75) / (num_90 - num_75)
                    predict_height = future_num_90 + alpha * (future_num_90 - future_num_75)
                else:
                    alpha = (height - num_90) / (num_95 - num_90)
                    predict_height = future_num_95 + alpha * (future_num_95 - future_num_90)

            else:
                # 当未来身高在源数据中找不到时，找后来的身高
                predict_height = res.ix[i+1, "babyHeight"]

        height_list.append(predict_height)
        # print height_list
        print "第" + str(i) + "次循环"

    future_height = np.array(height_list)
    height_data = pd.Series(future_height)
    height_data.to_csv('data/futureHeightData.csv', index=True)

    res["predictHeight"] = height_list
    res.to_csv('data/resultPredictHeightData.csv', index=False)

if __name__ == "__main__":
    main()