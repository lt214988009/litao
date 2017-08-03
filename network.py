# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:59:07 2017

@author: dell
"""
import networkx as nx
import matplotlib.pyplot as plt
import pylab 
import numpy as np
#自定义网络
row=np.array([0,0,0,1,2,3,6])
col=np.array([1,2,3,4,5,6,7])
value=np.array([1,2,1,8,1,3,5])
print('生成一个空的有向图')
G=nx.DiGraph()
print('为这个网络添加节点...')
for i in range(0,np.size(col)+1):
    G.add_node(i)
print('在网络中添加带权中的边...')
for i in range(np.size(row)):
    G.add_weighted_edges_from([(row[i],col[i],value[i])])
print('输出网络中的节点...')
print(G.nodes())
print('输出网络中的边...')
print(G.edges())
print('输出网络中边的数目...')
print(G.number_of_edges())
print('输出网络中节点的数目...')
print(G.number_of_nodes())
print('给网路设置布局...')
pos=nx.shell_layout(G)
print('画出网络图像：')
nx.draw(G,pos,with_labels=True, node_color='green', edge_color='red', node_size=400, alpha=0.5 )
pylab.title('Self_Define Net',fontsize=15)
#pylab.show()
l=[]
klist =list(nx.k_clique_communities(G,3)) #list of k-cliques in the network. each element contains the nodes that consist the clique.

#plotting
pos = nx.spring_layout(G)
plt.clf()
nx.draw(G,pos = pos, with_labels=False)
nx.draw(G,pos = pos, nodelist = l[0], node_color = 'b')
nx.draw(G,pos = pos, nodelist = l[1], node_color = 'y')
#plt.show()




import networkx as nx
G=nx.random_graphs.barabasi_albert_graph(1000,3)   #生成一个n=1000，m=3的BA无标度网络
print (G.degree(0))                                   #返回某个节点的度
print (G.degree())                                     #返回所有节点的度
print (nx.degree_histogram(G))    #返回图中所有节点的度分布序列（从1至最大度的出现频次）
import matplotlib.pyplot as plt                 #导入科学绘图的matplotlib包
degree =  nx.degree_histogram(G)          #返回图中所有节点的度分布序列
x = range(len(degree))                             #生成x轴序列，从1到最大度
y = [z / float(sum(degree)) for z in degree]  
#将频次转换为频率，这用到Python的一个小技巧：列表内涵，Python的确很方便：）
plt.loglog(x,y,color="blue",linewidth=2)           #在双对数坐标轴上绘制度分布曲线  
plt.show()                                                          #显示图表
nx.average_clustering(G)#群聚集系数
nx.clustering(G) #所有节点的聚集系数
nx.diameter(G)#返回图G的直径（最长最短路径的长度）
nx.average_shortest_path_length(G)#则返回图G所有节点间平均最短路径长度
nx.degree_assortativity(G)#可以计算一个图的度匹配性
nx.degree_centrality(G)#Compute the degree centrality for nodes.
nx.in_degree_centrality(G)#Compute the in-degree centrality for nodes.
nx.out_degree_centrality(G)#Compute the out-degree centrality for nodes


eigenvector_centrality(G[, max_iter, tol, ...])
#Compute the eigenvector centrality for the graph G.
eigenvector_centrality_numpy(G)#Compute the eigenvector centrality for the graph G
load_centrality(G[, v, cutoff, normalized, ...])##Compute load centrality for nodes.
edge_load(G[, nodes, cutoff]) ##Compute edge load.




#定义一个方法，它有两个参数：n - 网络节点数量；m - 每步演化加入的边数量
def barabasi_albert_graph(n, m):
    # 生成一个包含m个节点的空图 (即BA模型中t=0时的m0个节点) 
    G=empty_graph(m)  
    # 定义新加入边要连接的m个目标节点
    targets=range(m)  
    # 将现有节点按正比于其度的次数加入到一个数组中，初始化时的m个节点度均为0，所以数组为空 
    repeated_nodes=[]     
    # 添加其余的 n-m 个节点，第一个节点编号为m（Python的数组编号从0开始）
    source=m 
    # 循环添加节点
    while source        # 从源节点连接m条边到选定的m个节点targets上（注意targets是上一步生成的）
        G.add_edges_from(zip([source]*m,targets)) 
        # 对于每个被选择的节点，将它们加入到repeated_nodes数组中（它们的度增加了1）
        repeated_nodes.extend(targets)
        # 将源点m次加入到repeated_nodes数组中（它的度增加了m）
        repeated_nodes.extend([source]*m) 
        # 从现有节点中选取m个节点 ，按正比于度的概率（即度优先连接）
        targets=set()
        while len(targets)            #按正比于度的概率随机选择一个节点，见注释1
            x=random.choice(repeated_nodes) 
            #将其添加到目标节点数组targets中
            targets.add(x)        
        #挑选下一个源点，转到循环开始，直到达到给定的节点数n
        source += 1 
    #返回所得的图G
    return G



import networkx as nx                   #导入networkx包
import matplotlib.pyplot as plt     #导入绘图包matplotlib（需要安装，方法见第一篇笔记）
G =nx.random_graphs.barabasi_albert_graph(100,1)   #生成一个BA无标度网络G
nx.draw(G)                          #绘制网络G
plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件

'''
 - `node_size`:  指定节点的尺寸大小(默认是300，单位未知，就是上图中那么大的点)
      - `node_color`:  指定节点的颜色 (默认是红色，可以用字符串简单标识颜色，例如'r'为红色，'b'为绿色等，具体可查看手册)
      - `node_shape`:  节点的形状（默认是圆形，用字符串'o'标识，具体可查看手册）
      - `alpha`: 透明度 (默认是1.0，不透明，0为完全透明) 
      - `width`: 边的宽度 (默认为1.0)
      - `edge_color`: 边的颜色(默认为黑色)
      - `style`: 边的样式(默认为实现，可选： solid|dashed|dotted,dashdot)
      - `with_labels`: 节点是否带标签（默认为True）
      - `font_size`: 节点标签字体大小 (默认为12)
      - `font_color`: 节点标签字体颜色（默认为黑色）
'''










