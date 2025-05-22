import dgl
import torch
import numpy as np
from dgl.data import DGLDataset


class ChangeDataset(DGLDataset):
    def __init__(self, name, data_path="./dataset"):
        self.data_path = data_path
        super().__init__(name=name)

    def process(self):
        # 读取数据
        adj_matrix = np.load(f"{self.data_path}/{self.name}/{self.name}_adj.npy")
        features = np.load(f"{self.data_path}/{self.name}/{self.name}_feat.npy")
        labels = np.load(f"{self.data_path}/{self.name}/{self.name}_label.npy")

        # adj_matrix = np.load(f"{self.data_path}/dblp/dblp_adj.npy")
        # features = np.load(f"{self.data_path}/dblp/dblp_feat.npy")
        # labels = np.load(f"{self.data_path}/dblp/dblp_label.npy")

        # 创建 DGLGraph
        src, dst = np.nonzero(adj_matrix)
        self.graph = dgl.graph((src, dst), num_nodes=adj_matrix.shape[0])

        # 设置节点特征
        features = np.array(features, dtype=np.float32)  # 转换为 float32

        self.graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32)
        self.graph.ndata['label'] = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1  # 只有一个图

    def num_nodes(self):
        return self.graph.num_nodes()

    def num_classes(self):
        return len(torch.unique(self.graph.ndata['label']))


# GRAPH_DICT 模拟不同数据集
GRAPH_DICT = {
    "acm": lambda: ChangeDataset("acm")
}
