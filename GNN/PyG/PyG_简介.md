+ [项目地址](https://github.com/rusty1s/pytorch_geometric)
+ [Document](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

PyTorch Geometric (PyG)是pytorch的图神经网络库，大大简化了实现图卷积网络的过程。

### Message Passing

将卷积操作扩展到无规则数据结构领域，可以表示为`neighborhood aggregation`或者`message passing`模式。$\mathbf{x}^{(k-1)}_i \in \mathbb{R}^F$表示在$(k-1)$层的第$i$个节点的特征向量，$\mathbf{e}_{i,j} \in \mathbb{R}^D$(optional)表示从节点$i$到节点$j$的边的特征向量，MPGNN可以表示为
$$
\mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{i,j}\right) \right),
$$
其中，$\square$表示可导的、与输入顺序无关的函数，比如求和、求平均或取最大值，$\gamma$和$\phi$表示可导的函数，比如MLP(多层感知机)。

### MessagePassing 基类

PyG提供`torch_geometric.nn.MessagePassing`基类，用于创建这种MPGNN模型，构建自定义的MPNN时，只需要定义函数$\phi$(即`message()`)和$\gamma$(即`update()`)，以及信息聚合模式，比如`aggr='add'`，`aggr='mean'`或者`aggr='max'`。

PyG通过如下方法实现上述功能：

+ `torch_geometric.nn.MessagePassing(aggr="add",flow="source_to_target")`定义了聚合模式(`aggr`)以及信息的流动方向(`flow`)。
+ `torch_geometric.nn.MessagePassing.propagate(edge_index,size=None,**kwargs)`函数输入边的indices和所有需要的数据，用于构建信息以及对节点的embedding进行更新。
+ `torch_geometric.nn.MessagePassing.message()`类似于函数$\phi$构建传递到节点$i$的信息，能够接受任何传递给`propagate()`函数的参数，另外，可以通过在节点$i$后加入后缀`_i`来将特征映射到对应的节点，比如使用`x_i`表示节点$i$的特征向量。
+ `torch_geometric.nn.MessagePassing.update()`用于对每个节点$i \in \mathcal{V}$以类似于函数$\gamma$的方式来更新节点的embedding，该函数接受信息聚合函数的输出作为第一个输入参数，以及任意的最初传递给`propagate()`的参数。

#### 实现GCN Layer

GCN层定义如下
$$
\mathbf{x}_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{deg(j)}} \cdot \left( \mathbf{\Theta} \cdot \mathbf{x}_j^{(k-1)} \right),
$$
其中，将邻居节点特征先通过权重矩阵$\mathbf{\Theta}$进行变换，然后使用节点的度进行归一化，最后再求和，这个公式可以分解为以下步骤：

1. 增加自连接到邻接矩阵，即邻接矩阵的对角线元素为1。
2. 对节点的特征矩阵进行线性变换。
3. 使用函数$\phi$对节点特征进行规范化。
4. 对邻居节点特征进行聚合操作。
5. 通过函数$\gamma$返回新的节点embedding



整个实现的代码如下：

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]
        # edge_index has shape [2, E]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)  # [N, ]
        deg_inv_sqrt = deg.pow(-0.5)   # [N, ]
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out
```

该网络`GCNConv`从类`torch_geometric.nn.MessagePassing`进行继承，使用`add`的聚合方式，网络每层之间的操作在`forward()`函数里面进行实现，在`forward()`函数中，首先增加节点的自连接到边列表，使用函数`torch_geometric.utils.add_self_loops()`，然后使用线性变换函数对节点特征进行变换。之后再调用`propagate()`函数，在该函数内部会调用`message()`函数和`update()`函数，分别进行信息产生以及更新操作。

在`message()`函数中，需要实现邻居节点的特征向量$x_j$的归一化，其中，$x_j$包含了每一条边对应邻居节点的特征向量，节点的特征能够通过在变量名后添加后缀`_i`和`_j`自动地映射得到。

邻居节点特征通过计算$i$节点的度$deg(i)$来进行归一化，并且将每一条边$(i,j) \in \mathcal{E}$的归一化数值保存在变量`norm`中。

在`update()`函数中，直接返回信息聚合的输出。