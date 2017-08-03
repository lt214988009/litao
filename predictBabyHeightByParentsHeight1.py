# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from areaComputeAlgorithm import compute
import clearSourceData
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import mode
import re

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
    print frame

    # 将父母身高分区间
    group_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    fa_height = frame["fatherHeight"]
    mo_height = frame["motherHeight"]
    print frame["fatherHeight"].min()
    print frame["fatherHeight"].max()
    print frame["motherHeight"].min()
    print frame["motherHeight"].max()
    bins_father = [fa_height.min(), 160, 165, 170, 175, 180, 185, fa_height.max()]
    print bins_father
    bins_mother = [mo_height.min(), 145, 150, 155, 160, 165, 170, mo_height.max()]
    print bins_mother
    fa_height_data = pd.cut(fa_height, bins_father, labels=group_names)
    print fa_height_data
    mo_height_data = pd.cut(mo_height, bins_mother, labels=group_names)
    print mo_height_data
    frame.insert(1, 'faHeight', fa_height_data)
    frame.insert(1, 'moHeight', mo_height_data)
    print frame

    # 统计每个区间下的用户数量
    area_frame = frame.groupby(["faHeight", "moHeight"]).count()
    area_frame = area_frame["fatherHeight"]
    print area_frame
    area_frame.to_csv("data/parentsHeightArea.csv")
    area_x = [i for i in range(len(area_frame))]
    area_y = area_frame
    plt.figure(0)
    plt.bar(area_x, area_y)
    plt.xlabel("area")
    plt.ylabel("number")
    plt.title("Statistic Number Per Area")
    plt.legend()
    plt.savefig('image/statisticNumberPerArea.png')

    # 统计每个年龄下的用户数量
    num_frame = frame["fatherHeight"].groupby(frame["babyAge"]).count()
    num_x = num_frame.index
    num_y = num_frame
    plt.figure(1)
    plt.bar(num_x, num_y)
    plt.xlabel("baby age")
    plt.ylabel("number")
    plt.title("Statistic Number Per Age")
    plt.legend()
    plt.savefig('image/statisticNumberPerAge.png')

    # 选择父亲身高区间为170-175与母亲身高区间为155-160的数据
    most_data = frame[
        (frame["fatherHeight"] >= 170) & (frame["fatherHeight"] <= 175) & (frame["motherHeight"] >= 155) & (
            frame["motherHeight"] <= 160)]

    # 删除父母身高字段
    most_data.drop("fatherHeight", 1)
    most_data.drop("motherHeight", 1)
    print most_data
    print most_data.dtypes
    data_frame = pd.DataFrame(most_data, columns=["babyHeight", "babyAge"])
    grouped = data_frame["babyHeight"].groupby(data_frame["babyAge"])

    # 计算分位数
    num_25 = grouped.quantile(0.25)
    num_50 = grouped.quantile(0.50)
    num_75 = grouped.quantile(0.75)
    num_100 = grouped.quantile(1.00)
    mean = grouped.mean()
    std = grouped.std()
    std = std.fillna(0)
    print std

    #df = pd.DataFrame([num_25, num_50, num_75, num_100, mean, std])
    df = pd.DataFrame([mean-2*std, mean-std, mean, mean+std, mean+2*std])
    df = df.T
    df.columns = ["mean-2*std", "mean-std", "mean", "mean+std", "mean+2*std"]

    # 保存清洗干净的数据到本地
    print df
    df.to_csv('data/clearedMeanData.csv', index=False)
    x = df.index

    # 将结果想成图表，并保存到本地
    plt.figure(2)
    plt.plot(x, num_25, label="$per$_$25$", color="green", linewidth=0.5)
    plt.plot(x, num_50, label="$per$_$50$", color="blue", linewidth=0.5)
    plt.plot(x, num_75, label="$per$_$75$", color="cyan", linewidth=0.5)
    plt.plot(x, num_100, label="$per$_$100$", color="red", linewidth=0.5)
    plt.xlabel("baby age")
    plt.ylabel("baby height")
    plt.title("Predict Height By Parents")
    plt.legend()
    plt.savefig('image/predictHeightByParents.png')

    # 预测未出生的宝宝身高图,标准差
    plt.figure(3)
    plt.plot(x, mean, label="$mean$", color="red", linewidth=0.5)
    plt.plot(x, mean + std, label="$mean+std$", color="green", linewidth=0.5)
    plt.plot(x, mean - std, label="$mean-std$", color="blue", linewidth=0.5)
    plt.plot(x, mean + 2*std, label="$mean+2std$", color="cyan", linewidth=0.5)
    plt.plot(x, mean - 2*std, label="$mean-2std$", color="yellow", linewidth=0.5)
    plt.xlabel("baby future age")
    plt.ylabel("baby future height")
    plt.title("Predict future Height By Parents")
    plt.legend()
    plt.savefig('image/predictFutureHeightByParents.png')

    # 根据每个区间下的用户数量绘制热图
    area_atrr = pd.read_csv("data/parentsHeightAtrr.csv")
    area_data = np.array(area_atrr)
    print area_data
    rows = list('1234567')
    columns = list('1234567')
    fig, ax = plt.subplots()
    ax.pcolor(area_data, cmap=plt.cm.Reds, edgecolors='K')
    ax.set_xticks(np.arange(0, 7) + 1.0)
    ax.set_yticks(np.arange(0, 7) + 1.0)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.set_xticklabels(columns, minor=False, fontsize=20)
    ax.set_yticklabels(rows, minor=False, fontsize=20)
    plt.savefig('image/statisticNumberPerAreaByHeat.png')
    plt.show()

if __name__ == "__main__":
    main()