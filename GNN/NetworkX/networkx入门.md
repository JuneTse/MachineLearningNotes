## 简介

NetworkX诞生于2002年，是一个复杂网络建模的python包，用于创建、操作和研究复杂网络结构、动态和功能。NetworkX提供了：

* 研究社交、生物和基础设施网络的结构和动态的工具
* 标准的编程接口和图实现，适用于很多应用
* 协作、多学科交叉工程的快速开发环境
* 多种已有的接口
* 轻松处理非标准数据集的能力

使用NetworkX可以加载和存储标准化和非标准化的网络，生成随机和经典的网络，分析网络结构，创建网络模型，设计新的网络算法，绘制网络等。



## Tutorial

### Creating a graph

*Graph*: 图是节点和边的集合，在NetworkX中，节点可以是任何hashable的对象，比如文本字符串，图像，Graph，节点对象等。

创建一个无节点和边的空的图：

```python
import networkx as nx

G=nx.Graph()
```

* 可以使用多种不同的方式向图中增加节点和边，NetworkX包含多种图生成器函数，方便读取和写入多种格式的图。

```python
#向图加一个节点
G.add_node(1)

#向图中加多个节点，可以使用任意的可迭代容器：list等
G.add_nodes_from([2,3,4])
```

* 容器中的节点元素可以是一个包含节点属性的元组

```python
# 把另一个图中的节点加入到G中
H=nx.path_graph(10)
G.add_nodes_from(H)
# 也可以把图H作为一个节点
G.add_node(H)
```

### Edges

```python
#向图中增加一个边
G.add_edge(1,2)
e=(2,3)
G.add_edge(*e)

#增加多个边
G.add_edges_from([(1,2),(1,3)])
```

* 还可以使用ebunch（networkx中的一种容器）添加边，ebunch是包含edge-tuple的可迭代容器。一个edge可以是一个2-tuple的节点对，或者3-tuple的2个节点和一个边属性的字典，比如``` (2,3,{'weight':3.1415})```

```python
#把另一个图中的边加入到G中
G.add_edges_from(H.edges)
```

* 图中节点和边的个数

```python
#节点的个数
G.number_of_nodes()
#边的个数
G.number_of_edges()
```

* 图的属性信息

```python
#节点
nodes=list(G.nodes)
#边
edges=list(G.edges)
#度
degrees=G.degree
#邻接矩阵
G.adj()
```

### 删除图中的节点

删除操作和添加操作类似

```python
#删除一个节点
G.remove_node(2)
#删除多个节点
G.remove_nodes_from("spam")
#删除边
G.remove_edge(1,3)
```

### 访问边和邻居

可以使用```Graph.edges```和```Graph.adj()```的下标访问边和邻居

```python
# 访问邻居
G[1]  #等价于：G.adj[1]
#两个节点的边
G[1][2]
G.edges[1,2]
```

可以使用下标获取和设置边的属性

```python
G.add_edge(1,3)
G[1][3]['color']="blue"
G.edges[1,3]['color']='red'
```

使用```G.adjacency```或```G.adj.items()```访问图的邻接矩阵，注意对于无向图，```G.adjacency()```会访问每条边两次

```python
FG=nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
for n,nbrs in FG.adj.items():
    for nbr,eattr in nbrs.items():
        wt=eattr['weight']
        if wt <0.5: print(f"({n}, {nbr}, {wt:.3})")
```

### 为图，节点和边增加属性

属性，比如权重、标签、颜色和任何其他python对象，都可以添加到图、节点和边上。



### 有向图

* ```DiGraph```类专门为有向图提供了额外的属性，比如```DiGraph.out_edges()```,```DiGraph.in_degree```,```DiGraph.predecenssors()```,```DiGraph.successors()```等。
* 为了使算法能够在两种图上都工作，有向图的```neighbors()```等价于```successors()```,```degree```则表示```in_degree```和```out_degree```的和，即使这样看起来不一致。
* 可以使用```Graph.to_undirected()```转换成无向图

### 图生成器和图操作

常见的图操作

```python
subgraph(G,nbunch) # 节点为nbunch的子图
union(G1,G2) #图并集
...
```

生成一些典型的图

```python
petersen=nx.petersen_graph()
tutte=nx.tutte_graph()
...
```

从文件中读写图

```python
nx.write_gml(red,"path.to.file")
mygraph=nx.read_gmx("path.to.file")
```



### 分析图

* 可以使用多种图理论函数分析图的结构，比如```nx.clustering(G)```和```nx.connected_components(G)```

* ```nx.all_pairs_shortest_path(G)```

### 绘制图

NetworkX不是专门的绘制图的包，但是包含了一些使用Matplotlib和Graphviz的基本绘制接口。这些是```networkx.drawing```模块的一部分。

首先，导入matplotlib的绘制接口

```python
import matplotlib.pyplot as plt
```

绘制图

```python
G=nx.petersen_graph()
plt.subplot(121)
nx.draw(G,with_labels=True,font_weight="bold")
plt.subplot(122)
nx.draw_shell(G,nlist=[range(5,10),range(5)],with_labels=True,font_weight="bold")
```









