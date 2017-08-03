，import pandas as pd
import numpy as np
import os
from kmodes import kmodes
import shutil


def read_file(surveyid):
    # 读取清洗后的数据文件，清洗后的数据文件变量排序是
    # 问题1 ... 问题N，是否有改善，改善方法1 ... 改善方法N
    # filename 目录名称
    # features 问题1 ... 问题N
    # effect 是否有改善 0 = 改善；1 = 不变； 2 = 恶化
    # targets 改善方法1 ... 改善方法N

    filename = "survey56_cluster.csv"
    clean_df = pd.read_csv(filename, sep=',', na_values='NA')
    features = clean_df.iloc[:, :99]
    effect = clean_df['qid240']
    targets = clean_df.iloc[:, 102:113]
    return clean_df, features, effect, targets


def type1_rate(targets, labels, effect, n_clusters, usableStd=0):
    # 聚类指标计算不需要变动
    total_better = 0
    total_worse = 0
    total_all = 0
    for i in range(n_clusters):
        clstr_i = targets.loc[labels == i, ]
        better = clstr_i.loc[effect == 0, ].sum(axis=0, skipna=True)
        worse = clstr_i.loc[effect == 2, ].sum(axis=0, skipna=True)
        total = clstr_i.sum(axis=0, skipna=True)
        total.loc[total < usableStd] = float("inf")

        percent = (better - worse) / total
        percent.sort_values(inplace=True, ascending=False)
        top_idx = percent.index[0]  # Select best method in cluster_i

        total_better += better.loc[top_idx]
        total_worse += worse.loc[top_idx]
        total_all += total.loc[top_idx]

    return (total_better - total_worse) / total_all


def cluster_similarity(features, center, labels):
    # 聚类指标计算不需要变动
    n_clusters, base = center.shape
    avg_similarity = []
    for i in range(n_clusters):
        clstr_i = np.array(features.loc[labels == i, ])
        cost = np.sum(clstr_i != center[i, ], axis=1).mean()
        similariy = (base - cost) / base
        avg_similarity.append(similariy)
    total_similarity = sum(avg_similarity) / len(avg_similarity)
    return total_similarity


def predict_labels(features, centroids):
    # 预测分类函数不需要变动
    features = np.array(features)
    labels_tmp_shape = (features.shape[0], centroids.shape[0])
    labels_tmp = np.empty(labels_tmp_shape, dtype=np.uint8)
    for ipoint, curpoint in enumerate(centroids):
        labels_tmp[:, ipoint] = np.sum(features != curpoint, axis=1)
    labels = np.argmin(labels_tmp, axis=1)
    return labels


def labels_description(labels):
    # 描述分类结果函数不需要变动
    labels_count = [0] * np.unique(labels)
    for i in labels:
        labels_count[i] += 1
    labels_count = np.array(labels_count)
    return labels_count.mean(), labels_count.std()


def cluster_centroid_finder(features, effect, targets,
                            init='cao', turn=5, rangefree=False,
                            rangelow=2, rangehigh=10,
                            savefile=True, savecenter=True, surveyid=14):
    # 聚类中心数量探索函数不需要变动
    type1 = []
    centers = []
    labels = []
    similariy = []

    if rangefree is not False:
        rangecase = rangefree
    else:
        rangecase = range(rangelow, rangehigh + 1)

    for n_clusters in rangecase:
        print("n_cluster: ", n_clusters)
        km = kmodes.KModes(n_clusters=n_clusters,
                           init=init,
                           max_iter=turn, verbose=True)
        km.fit_predict(features)
        type1_score = type1_rate(targets, np.array(km.labels_), effect, n_clusters)
        type1.append(type1_score)
        centers.append(km.cluster_centroids_)
        labels.append(km.labels_)
        similariy_score = cluster_similarity(features, km.cluster_centroids_, km.labels_)
        similariy.append(similariy_score)

    if savefile:
        with open("type1.csv", 'w') as fp:
            type1_str = map(lambda x: str(x), type1)
            similariy_str = map(lambda x: str(x), similariy)
            fp.write(',type1,similarity\n')
            for line in zip(rangecase, type1_str, similariy_str):
                line = [str(ele) for ele in line]
                fp.write(','.join(line) + '\n')

    if savecenter:
        path = "survey" + str(surveyid) + "centers"
        isExists = os.path.exists(path)
        if isExists:
            shutil.rmtree(path)
        os.makedirs(path)
        for idx, cts in zip(rangecase, centers):
            np.savetxt(path + '/cts_' + str(idx), cts, fmt='%d', delimiter=',')
        with open(path + '/cts_names', 'w') as fp:
            fp.write(','.join(features.columns))

    best_idx = np.array(type1).argmax()
    best_idx1 = np.array(similariy).argmax()
    print("Best number of Clusters on type1:",list(rangecase)[best_idx])
    print("Best number of Clusters on similarity:",list(rangecase)[best_idx1])
    # return type1, centers, similariy


def summary(surveyid, ctsid, targets, effect, reportpath):
    # 聚类中心坐标探索函数不需要变动
    cts_path = "survey" + str(surveyid) + "centers/cts_" + str(ctsid)
    centroids = np.genfromtxt(cts_path, dtype='float', delimiter=',')

    labels = predict_labels(features, centroids)
    pd.DataFrame(np.transpose(np.matrix(labels))).to_csv(
        reportpath + "/labels.csv", header=None, index=None)

    targets['effect'] = effect
    targets.loc[effect == 0, 'effect'] = "Better"
    targets.loc[effect == 1, 'effect'] = "NoChange"
    targets.loc[(effect == 2) | (effect == 3), 'effect'] = "Worse"
    targets['labels'] = labels
    result = targets.groupby(['labels', 'effect']).count()
    result.to_csv(reportpath + "/survey" + str(surveyid) + "_method.csv")

    summary = targets.loc[:, ('labels', 'effect')].groupby(['labels']).count()
    centroids = pd.DataFrame(centroids)
    with open("survey" + str(surveyid) + "centers/cts_names", 'r') as fp:
        centroids.columns = fp.readlines()[0].strip().split(',')
    pd.concat([summary, centroids], axis=1).to_csv(
        reportpath + "/survey" + str(surveyid) + "_cluster.csv")


if __name__ == '__main__':
    # MODE = 1 执行聚类中心数量搜索任务
    # MODE = 2 执行聚类中心坐标搜索任务
    MODE = 2
    # surveyid即问卷id，用于生成各项文件名称
    surveyid = 56

    # 生成用于储存计算结果的文件夹
    reportpath = "survey" + str(surveyid) + "report"
    isExists = os.path.exists(reportpath)
    if not isExists:
        os.makedirs(reportpath)

    clean_df, features, effect, targets = read_file(surveyid)
    # Select Center
    if MODE == 1:
        # 需要调整rangelow与rangehigh的数值，这两个数值代表搜索的上下界
        # 这里2与30代表搜索 聚为2类一直到聚为30类的结果
        # 如果想要自己确定搜索范围，需要修改rangefree，将False替换为想要探索的数值
        # 例如rangefree=(10, 50, 160)，如此就是探索这三个数值的聚类结果

        # 最后会计算不同聚类结果的得分，并输出最优结果
        cluster_centroid_finder(features, effect, targets,
                                rangefree=False,
                                rangelow=2, rangehigh=25,
                                
                                surveyid=surveyid)
								
    # Choose Center
    if MODE == 2:
        # 如果需要尝试同一种中心的不同聚类结果，可以在MODE1中设置rangelow=2，rangehigh=2，这样就是同一种中心
        # 通常来说确定了聚类中心数量后也就确定了聚类中心结果
        # ctsid就是最佳聚类中心数量
        ctsid = 8
        summary(surveyid, ctsid, targets, effect, reportpath)