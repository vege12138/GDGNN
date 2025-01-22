import math
from typing import Dict, List, Optional

from torch import Tensor
from torch.nn import LSTM, Linear

import opt
from utils import *
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
from torch_sparse import spmm

class FusionGCN(nn.Module):
    def __init__(self, num_nodes, num_node_features,num_classes, opt):
        super(FusionGCN, self).__init__()
        self.num_layers = opt.num_convs
        self.opt = opt
        self.vae = VAE(num_node_features, opt.hidden_dim * 2, opt.hidden_dim)

        self.fc = nn.Linear(opt.hidden_dim, num_classes)
        self.moe = SparseMoE(opt.hidden_dim, num_classes, opt.num_experts)

        self.alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)
        if opt.use_custom_bias:
            self.bias = nn.Parameter(torch.empty((num_nodes, opt.hidden_dim)).uniform_(-0.01, 0.01), requires_grad=True)
        else:
            self.bias = None

        self.fc1 = nn.Linear(num_node_features, opt.hidden_dim * 2)
        self.fc2 = nn.Linear(opt.hidden_dim* 2, opt.hidden_dim)



    def graph_convolution(self, tem_h1, normalized_adjacency_matrix):
        """
        Perform graph convolution with layer fusion.

        Args:
            tem_h (torch.Tensor): Initial node features.
            normalized_adjacency_matrix (torch.Tensor): Normalized adjacency matrix.
            num_layers (int): Number of layers to apply.
            use_bias (bool): Whether to add a trainable bias to the output.

        Returns:
            torch.Tensor: Fused node features after multiple graph convolutions.
        """
        layer_out = []
        for i in range(self.num_layers):
            tem_h1 = torch.spmm(normalized_adjacency_matrix, tem_h1)
            layer_out.append(tem_h1)

        # Initialize decay results
        n, _ = tem_h1.shape
        decay_results = torch.ones((n, self.num_layers), device=tem_h1.device)

        # Compute decay factors
        factor = 1
        for t in range(1, self.num_layers):
            factor *= F.tanh(self.beta)+1

            # factor *= self.beta
            #factor *= 1
            decay_results[:, t] = factor


        # Apply Softmax to decay factors
        softmax_results = torch.softmax(decay_results, dim=1).unsqueeze(-1)

        # Stack and fuse layers
        x = torch.stack(layer_out, dim=1)
        fused_h = (x * softmax_results).sum(dim=1)

        #self.bias.data = self.bias.data.clamp_(-0.001, 0.001)
        if self.bias is not None:
            fused_h = fused_h + F.tanh(self.bias)

        h2 = F.relu(fused_h)


        #return layer_out[-1]

        return h2

    def forward(self, data, normalized_adjacency_matrix):
        x, edge_index = data.ndata['feat'], torch.stack(data.edges())

        h1 = self.vae(x)

        # h1 = F.relu(self.fc1(x))
        # h1 = self.fc2(h1)


        # h1 = F.dropout(h1, p=self.opt.dropP, training=self.training) #TP
        # h1 = self.fc(h1)
        #
        # tem_h1 = h1
        # h2 = self.graph_convolution(tem_h1 , normalized_adjacency_matrix) + self.opt.oriRes * h1



        tem_h1 = h1.clone()

        h2 = self.graph_convolution(tem_h1 , normalized_adjacency_matrix) + self.opt.oriRes * h1
        h2 = F.dropout(h2, p=self.opt.dropP, training=self.training) #TPT

        h2 = self.moe(h2, self.opt.dropP)  # 使用 MoE 模块处理

        # 分类层
        h2 = F.log_softmax(h2, dim=1)

        return h2

class SparseMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])

        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            # nn.Linear(input_dim, 64),
            # nn.ReLU(),
            # nn.Linear(64, num_experts),
            # nn.Softmax(dim=1)  # Probability distribution over experts
        )

    def forward(self, x, drop):
        # 计算门控权重
        gate_scores = self.gate(x)  # [batch_size, num_experts]


        sparse_gate = F.softmax(gate_scores, dim=1)  # Softmax 归一化


        # 获取所有专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, num_experts, output_dim]

        sparse_gate = F.dropout(sparse_gate, p=drop, training=self.training)
        expert_outputs = F.dropout(expert_outputs, p=drop, training=self.training)

        # 按稀疏门控权重加权专家输出
        output = torch.einsum('be,bem->bm', sparse_gate, expert_outputs)  # [batch_size, output_dim]

        return output

class VAE(nn.Module):
    def __init__(self, input_dim=3703, hidden_dim=1024, latent_dim=512):
        super(VAE, self).__init__()

        # 编码器：降维到隐含层，然后分别输出均值和对数方差
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # 解码器：从隐含层升维回到原始输入维度
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # 输入 -> 隐含层，并应用激活函数
        x = F.normalize(x, p=2, dim=1)
        h = self.encoder_fc1(x)
        #F.dropout(h, p=0.1, training=self.training)

        h = F.relu(h)
        #F.dropout(h, p=0.5, training=self.training)
        # 隐含层分别映射到均值和对数方差
        mu = self.encoder_fc_mu(h)
        log_var = self.encoder_fc_log_var(h)

        return mu, log_var


    def reparameterize(self, mu, log_var):
        # 重参数化技巧
        std = torch.exp(0.5 * log_var)  # 计算标准差
        eps = torch.randn_like(std)  # 标准正态分布采样
        return mu + eps * std  # 生成隐变量 z

    def forward(self, x):
        # 编码过程：获取均值和对数方差
        mu, log_var = self.encode(x)

        # 使用重参数化技巧生成隐变量
        z = self.reparameterize(mu, log_var)
        z = F.normalize(z)

        return z

    def decode(self, z):
        # 隐变量 -> 隐含层，并应用激活函数
        h = F.relu(self.decoder_fc1(z))

        # 隐含层映射回原始输入维度
        return self.decoder_fc2(h)





