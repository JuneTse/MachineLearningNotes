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

class GCNConv(MessagePassing):
    def __init__(self,in_channels,out_channels):
        super(GCNConv,self).__init__(aggr="add") #聚集方式：add
        self.lin=torch.nn.Linear(in_channels,out_channels)
    def forward(self,x,edge_index):
        # x has shape [N, in_channels]
        #edge_index has shape [2, E]

        #step 1: add self-loops to the adjacency matrix
        edge_index,_=add_self_loops(edge_index,num_nodes=x.size(0))

        #step 2: linearly transform node feature matrix
        x=self.lin(x)

        #step 3： start propagating message
        messages=self.propagate(edge_index,size=(x.size(0),x.size(0)),x=x)
        return messages
    def message(self,x_j,edge_index,size):
        # x_j has shape [E,out_channels]
        # edge_index has shape [2,E]
        # Step 3: normalize node features
        row, col=edge_index
        deg=degree(row,size[0],dtype=x_j.dtype) #[N,]
        deg_inv_sqrt=deg.pow(-0.5)
        norm=deg_inv_sqrt[row]*deg_inv_sqrt[col]

        return norm.view(-1,1)*x_j
    def update(self,aggr_out):
        # aggr_out hash shape [N,out_channels]
        # Step 5: Return new node embeddings
        return aggr_out
class Net(torch.nn.Module):
    def __init__(self,in_dim,num_class):
        super(Net,self).__init__()
        self.conv1=GCNConv(in_dim,16)
        self.conv2=GCNConv(16,num_class)
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

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Net(in_dim,num_class).to(device)

    data=dataset[0].to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=5e-5)

    #训练
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out=model(data)
        loss=F.nll_loss(out[data.train_mask],data.y[data.train_mask])
        loss.backward()
        optimizer.step()
     #在测试上评估模型
    model.eval()
    _,pred=model(data).max(dim=1)
    correct=float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc=correct/data.test_mask.sum().item()
    print("Accuracy:{:.4f}".format(acc))





