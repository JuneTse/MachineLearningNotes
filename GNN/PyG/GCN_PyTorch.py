#coding:utf-8
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree,softmax

from torch_geometric.datasets import Planetoid

'''
实现信息聚合函数，将前面使用邻居信息按照每一个source节点进行聚合，
之后用于更新每一个节点的状态向量。
Initialize :
Input :
    node_num : (int)节点的数量
Forward :
Input :
    H : (Tensor)输入邻居向量，shape为(N, dim), N是边的个数
    X_node : (Tensor)H每一行对应source节点的索引，shape为(N, )
Output :
    out : (Tensor)求和式聚合之后的新的节点状态向量，shape为(V, dim)，V为节点个数
'''
class AggrSum(nn.Module):
    def __init__(self, node_num):
        super(AggrSum, self).__init__()
        self.V = node_num

    def forward(self, H, X_node):
        # H : (N, s) -> (V, s)
        # X_node : (N, )
        mask = torch.stack([X_node] * self.V, 0) #[V, N]
        # 每一个对应的一个source节点的邻居，source节点对应的邻居位置值变成0
        mask = mask.float() - torch.unsqueeze(torch.range(0, self.V - 1).float(), 1).to(H.device)
        mask = (mask == 0).float() #source节点对应的邻居位置变成1
        # (V, N) * (N, s) -> (V, s) 收集邻居信息
        return torch.mm(mask, H)

'''
用于实现GCN的卷积块。
Initialize :
Input :
    in_channel : (int)输入的节点特征维度
    out_channel : (int)输出的节点特征维度
Forward :
Input :
    x : (Tensor)节点的特征矩阵，shape为(N, in_channel)，N为节点个数
    edge_index : (Tensor)边矩阵，shape为(2, E)，E为边个数。
Output :
    out : (Tensor)新的特征矩阵，shape为(N, out_channel)
'''
class GCNConv(nn.Module):
    def __init__(self,in_channels,out_channels,node_num):
        super(GCNConv,self).__init__() #聚集方式：add
        self.lin=torch.nn.Linear(in_channels,out_channels)
        self.aggregation=AggrSum(node_num)

    def addSelfConnect(self,edge_index,num_nodes):
        selfconn=torch.stack([torch.range(0,num_nodes-1,dtype=torch.long)]*2,
                             dim=0).to(edge_index.device)
        return torch.cat(tensors=[edge_index,selfconn],dim=1)
    def calDegree(self,edges,num_nodes):
        ind,deg=np.unique(edges.cpu().numpy(),return_counts=True)
        deg_tensor=torch.zeros((num_nodes,),dtype=torch.long)
        deg_tensor[ind]=torch.from_numpy(deg)
        return deg_tensor.to(edges.device)
    def forward(self,x,edge_index):
        # x has shape [N, in_channels]
        #edge_index has shape [2, E]

        #step 1: add self-loops to the adjacency matrix
        edge_index=self.addSelfConnect(edge_index,num_nodes=x.shape[0])

        #step 2: linearly transform node feature matrix
        x=self.lin(x)

        #step 3： normalize message
        row,col=edge_index
        deg=self.calDegree(row,x.shape[0]).float()
        deg_sqrt=deg.pow(-0.5)
        norm=deg_sqrt[row]*deg_sqrt[col]

        #Node feature maxtrix
        tar_matrix=torch.index_select(x,dim=0,index=col)
        tar_matrix=norm.view(-1,1)*tar_matrix
        #Aggregate information
        aggr=self.aggregation(tar_matrix,row)
        return aggr

        return messages

class Net(torch.nn.Module):
    def __init__(self,in_dim,num_class,num_nodes):
        super(Net,self).__init__()
        self.conv1=GCNConv(in_dim,16,num_nodes)
        self.conv2=GCNConv(16,num_class,num_nodes)
    def forward(self,data):
        x,edge_index=data.x,data.edge_index

        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=F.dropout(x,training=self.training)
        x=self.conv2(x,edge_index)
        return F.log_softmax(x,dim=1)

if __name__=="__main__":
    dataset=Planetoid("tmp/Cora",name="Cora")
    print(dataset)
    in_dim=dataset.num_node_features
    num_class=dataset.num_classes
    num_nodes=len(dataset.data.x)




    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Net(in_dim,num_class,num_nodes=num_nodes).to(device)

    data=dataset[0].to(device)
    edge_index=data.edge_index
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=5e-5)

    #训练
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out=model(data)
        loss=F.nll_loss(out[data.train_mask],data.y[data.train_mask])
        print(loss)
        loss.backward()
        optimizer.step()
     #在测试上评估模型
    model.eval()
    _,pred=model(data).max(dim=1)
    correct=float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc=correct/data.test_mask.sum().item()
    print("Accuracy:{:.4f}".format(acc))





