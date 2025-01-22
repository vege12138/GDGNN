import random
import  pandas as pd
import networkx as nx
import numpy as np
import torch
from scipy import sparse
from scipy.sparse import coo_matrix

def save_results_to_excel(results_dict, filename):
    """
    Save the results dictionary to an Excel file.

    Args:
        results_dict (dict): A dictionary containing results for each run and the max mean accuracy.
        filename (str): The name of the Excel file to save.
    """
    # Prepare data for Excel
    rows = []
    for key, value in results_dict.items():
        if key == 'max_mean_acc':
            rows.append([key, value[0], None])  # Append max_mean_acc
        else:
            rows.append([key, value[0], value[1]])  # Append mean_acc and all_acc

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=["Run", "Mean_Accuracy", "All_Accuracies"])

    # Save to Excel
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")
def get_norm_matrix(adj):
    # 对称归一化矩阵
    adj = adj + coo_matrix(np.eye(adj.shape[0]))  # 添加自环

    # 计算度矩阵 D
    degree = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0  # 处理度为0的节点

    # 对称归一化：D^-1/2 * A * D^-1/2
    d_mat_inv_sqrt = coo_matrix(np.diag(d_inv_sqrt))
    normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # 转换为 COO 格式以提取 indices 和 values
    normalized_adj = normalized_adj.tocoo()

    # 转为 PyTorch 稀疏张量
    indices = torch.tensor([normalized_adj.row, normalized_adj.col], dtype=torch.long)
    values = torch.tensor(normalized_adj.data, dtype=torch.float32)
    shape = normalized_adj.shape
    torch_sparse_adj = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

    return torch_sparse_adj


def compare_layer_representations(x):
    """
    比较 x 的每一层与其他层的节点表示一致的个数。

    参数:
    x: torch.Tensor, 形状为 (num_nodes, num_layers, feature_dim)

    返回:
    layer_similarity: torch.Tensor, 形状为 (num_layers, num_layers)，
                      表示每层与其他层之间节点表示一致的个数。
    """
    num_nodes, num_layers, _ = x.shape
    layer_similarity = torch.zeros((num_layers, num_layers), dtype=torch.int32)

    for i in range(num_layers):
        for j in range(num_layers):
            if i != j:  # 跳过自身对比
                # 使用 torch.isclose 比较逐节点表示
                close_matrix = torch.isclose(x[:, i, :], x[:, j, :], atol=1e-5)
                # 逐节点检查表示是否完全一致
                same_nodes = close_matrix.all(dim=1)
                layer_similarity[i, j] = same_nodes.sum().item()  # 统计相同节点的数量

    return layer_similarity

def sample_adj_matrix(adj_matrix, sample_ratio=0.8):
    """
    对邻接矩阵进行随机采样，选择一定比例的边。
    """
    indices = adj_matrix["indices"]
    values = adj_matrix["values"]

    # 生成一个与边数相同的布尔掩码，以sample_ratio的概率选择边
    mask = torch.rand(indices.shape[1]) < sample_ratio

    # 仅保留被选中的边
    sampled_indices = indices[:, mask]
    sampled_values = values[mask]

    return {"indices": sampled_indices, "values": sampled_values}

def k_shell(graph, nums):
    g = graph.copy()
    importance_dict = {node: 0 for node in range(nums)}
    k = 1
    while g.number_of_nodes() > 0:
        # 找出当前度数小于等于 k 的节点
        level_node_list = [node for node, degree in g.degree() if degree <= k]

        # 如果当前 k 值没有节点符合条件，增加 k 值
        if not level_node_list:
            k += 1
            continue

        # 删除这些节点，并将它们的 k_shell 值更新为当前 k
        for node in level_node_list:
            importance_dict[node] = k

        # 从图中移除这些节点
        g.remove_nodes_from(level_node_list)

    return importance_dict

def F_score_helper(GT, found, common_elements):
    len_common = len(common_elements)
    precision = float(len_common)/len(found)
    if precision == 0:
        return 0

    recall = float(len_common)/len(GT)
    if recall == 0:
        return 0
    return (2*precision*recall)/(precision+recall)

def cal_F_score_helper(found_sets, GT_sets):
    d1 = {} #best match for an extracted community
    d2 = {} #best match for a known community

    for i in range(len(GT_sets)):
        gt = GT_sets[i]
        f_max = 0

        for j in range(len(found_sets)):
            f = found_sets[j]

            common_elements = gt.intersection(f)
            if len(common_elements) == 0:
                temp = 0
            else:
                temp = F_score_helper(gt, f, common_elements)

            f_max = max(f_max,temp)

            d1[j] = max(d1.get(j,0),temp)

        d2[i] = f_max

    return d1, d2

def process_set(sets):
    comms = []
    labels = set(sets)
    for label in labels:
        comm = set(np.where(sets == label)[0])
        comms.append(comm)
    return comms

def cal_F_score(found_sets, GT_sets, verbose=False):
    found_sets = process_set(found_sets)
    GT_sets = process_set(GT_sets)
    d1,d2 = cal_F_score_helper(found_sets, GT_sets)

    if d1 == None:
        return [0]*6

    vals1 = sum(d1.values())/len(d1)
    vals2 = sum(d2.values())/len(d2)
    f_score = vals1 + vals2
    f_score /= 2
    f_score = round(f_score,4)
    vals1 = round(vals1,4)
    vals2 = round(vals2,4)

    return f_score, vals1, vals2

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_adjacency_matrix(graph):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
    values = [1 for index in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((values, (index_1, index_2)),
                          shape=(node_count, node_count),
                          dtype=np.float32)
    return A

def create_propagator_matrix(graph):
    """
    Creating a propagator matrix.
    :param graph: NetworkX graph.
    :return propagator: Dictionary of matrix indices and values.
    """
    A = create_adjacency_matrix(graph)
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    propagator["indices"] = torch.LongTensor(ind.T).to(device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(device)
    return propagator

def get_node_degree(graph, total_nodes):
 # 确保传入图的副本，避免图被修改
    node_degree = {node: 0 for node in range(total_nodes)}

    for node in graph:
        node_degree[node] = len(graph[node])
    return node_degree
def pyg_data_to_nx_graph(data):
    # 将 PyTorch Geometric 的边索引转换为边列表
    if data.device != torch.device('cpu'):
        data = data.to('cpu')

    # 提取 DGLGraph 的边列表，并转换为 (source, target) 形式的 numpy 数组
    edge_index = torch.stack(data.edges()).numpy()
    edge_list = list(zip(edge_index[0], edge_index[1]))

    # 创建 networkx 图并添加边
    graph = nx.Graph()
    graph.add_edges_from(edge_list)

    # 删除自环边
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph