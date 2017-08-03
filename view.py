from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
def distance_1(x1,x2):
    dist=0
    for i in range(len(x1)):
        if(x1[i] == x2[i]):
            diff=0
        else:
            diff=1
        dist+=diff
    return dist
filename='survey56_cluster_8.csv'
data=pd.read_csv(filename)
#抽样展示,由于数据量太大
#data_len=len(data)
#y1=data.iloc[:,98]
#df_values=list(set(y1.values))
#df_choice=[]
#for i in range(len(df_values)):
#    y_1=data.loc[y1==i,]
#    df_index=np.random.choice(y_1.iloc[:,0],size=int(len(y_1)*0.005),replace=False)
#    df_choice.append(y_1.iloc[df_index,:])
#data1=pd.DataFrame(df_choice)
X=data.iloc[0:1000,0:98]
y=data.iloc[0:1000,98:]
print('Computer t-sne embedding')
tsne=TSNE(n_components=2,init='pca',random_state=0,metric=distance_1)
X_tsne=tsne.fit_transform(X)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:,0], X_tsne[:, 1], c=y)
plt.savefig('figure_tsne1.pdf')
plt.show()
