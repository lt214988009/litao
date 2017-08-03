# -*-coding:utf-8 -*-
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl

def getData():
    # 读入源数据csv文件
    data = []
    with open('data/131.csv', 'r+') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append(row)
            '''
            for col in range(len(row)):
                if row[col] == '':
                    row[col] = 'NaN'
            data.append(row)
'''
    # 获取相关字段
    frame = pd.DataFrame(data)
    print(frame.head())
    frame = frame.drop_duplicates()

    frame = frame.loc[:, [0, 6, 10, 26, 32, 66, 69, 70]]  # 获取相关字段
    frame.columns = ["uID", "babyIsBirth", "babyBirthTime", "fatherHeight", "motherHeight",
                     "babyHeight", "babyAge", "submitTime"]

    # 删除babyHeight为空的行
    frame["babyHeight"] = frame["babyHeight"].replace('', 0)
    frame = frame[frame["babyHeight"] > 0]

    # 如果babyIsBirth字段为空，则用1填充（1表示宝宝未出生）
    frame["babyIsBirth"] = frame["babyIsBirth"].replace('', 1)

    # 如果父母身高字段为空，则用众数填充
    frame["fatherHeight"] = frame["fatherHeight"].replace('', mode(frame["fatherHeight"]).mode[0])
    frame["motherHeight"] = frame["motherHeight"].replace('', mode(frame["motherHeight"]).mode[0])

    # 如果babyBirthTime字段为空，则用0填充,并将其babyIsBirth字段设置为1
    frame["babyBirthTime"] = frame["babyBirthTime"].replace('', 0)

    # 筛选有效数据,删除未出生宝宝和宝宝出生时间为空的数据行
    frame = frame[(frame["babyIsBirth"] != 1) & (frame["babyBirthTime"] > 0) & (frame["babyBirthTime"] != 'NaN')]

    # 数据类型转换
    frame["submitTime"] = frame["submitTime"].astype("float64")
    frame["babyBirthTime"] = frame["babyBirthTime"].astype("float64")
    frame["fatherHeight"] = frame["fatherHeight"].astype("int")
    frame["motherHeight"] = frame["motherHeight"].astype("int")
    frame["babyHeight"] = frame["babyHeight"].astype("int")

    # 获取babyAge值，以每周为单位
    for index in frame.index:
        frame["babyAge"] = (frame["submitTime"] - frame["babyBirthTime"]) / (1000 * 3600 * 24 * 7)

    frame["babyAge"] = frame["babyAge"].astype("int")

    # 删除不相关字段，只保留父母身高和宝宝身高年龄字段
    del frame["uID"]
    del frame["babyIsBirth"]
    del frame["babyBirthTime"]
    del frame["submitTime"]

    x = frame.index
    x_age = frame["babyAge"]
    y = frame["fatherHeight"]
    z = frame["motherHeight"]
    k = frame["babyHeight"]

    # 对父母身高字段做数据探查
    plt.figure(1)
    ax = plt.subplot(121)
    plt.scatter(x, y)
    ax.set_title('Father Height')
    ax = plt.subplot(122)
    plt.scatter(x, z)
    ax.set_title('Mother Height')

    # 对宝宝身高字段做数据探查
    plt.figure(3)
    plt.scatter(x, k)
    plt.xlabel("index")
    plt.ylabel("height")
    plt.title("Baby Height Scatter")
    plt.legend()

    # 探测宝宝身高随年龄的散点分布图
    plt.figure(4)
    plt.scatter(x_age, k)
    plt.xlabel("baby age")
    plt.ylabel("baby height")
    plt.title("Baby Height By Age")
    plt.legend()

    # 保存可视化结果到本地
    plt.savefig('image/parentsHeightScatter.png')
    plt.savefig('image/babyHeightScatter.png')

    # 保存清洗干净的数据到本地
    frame.to_csv('data/clearedSourceData.csv', index=False)
