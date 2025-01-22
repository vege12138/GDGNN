import os
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from dgl.data import (
    CoraGraphDataset,
    CiteseerGraphDataset,
    PubmedGraphDataset,
    CoauthorCSDataset,
    AmazonCoBuyPhotoDataset,
    CoauthorPhysicsDataset,
    ChameleonDataset,
    SquirrelDataset, ActorDataset, WikiCSDataset, AmazonCoBuyComputerDataset,
)

from model import  FusionGCN
from torch_sparse import spmm

from opt import OptInit
from utils import *
from scipy.sparse import coo_matrix


def train():
    cnt = 0
    for epoch in range(opt.epochs):
        model.train()
        if epoch == 150:
            a = 1

        #dropedge
        # if epoch%20 == 0:
        #     normalized_adjacency_matrix = randomedge_sampler_norm(adj, 0.9).to(device)

        out, kl = model(data, normalized_adjacency_matrix)
        loss = criterion(out[data.train_mask], data.ndata['label'][data.train_mask]) + opt.kl * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

        _, pred = torch.max(out[data.train_mask], dim=1)
        correct = (pred == data.ndata['label'][data.train_mask]).sum().item()
        acc = correct / data.train_mask.sum().item()
        #
        # print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
        #     epoch, loss.item(), acc))

        val_loss, val_acc = valid()

        if record_dict['best_val_acc'] < val_acc :
            record_dict['best_epoch'] = epoch
            record_dict['best_val_acc']  = val_acc
            record_dict['best_test_acc'] = test()
            cnt = 0
        else:
            cnt += 1
        # print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f} || val_loss: {:.4f} val_acc: {:.4f}'.format(
        #     epoch, loss.item(), acc, val_loss, val_acc))
        if epoch % 10 == 0:
            print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f} || val_acc: {:.4f} ||  test_acc: {:.4f}'.format(
                epoch, loss.item(), acc, val_acc, record_dict['best_test_acc']))


        #early_stop次验证集没有提升就停止训练
        if cnt > opt.early_stop and opt.if_early_stop:
            a=1
            break




def valid():
    model.eval()
    with torch.no_grad():
        out, kl = model(data, normalized_adjacency_matrix)
        loss = criterion(out[data.val_mask], data.ndata['label'][data.val_mask])+ opt.kl * kl

        _, pred = torch.max(out[data.val_mask], dim=1)
        correct = (pred == data.ndata['label'][data.val_mask]).sum().item()
        acc = correct / data.val_mask.sum().item()


        return loss.item(), acc
        # print("val_loss: {:.4f} val_acc: {:.4f}".format(loss.item(), acc))


def test():
   # model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    out, kl = model(data, normalized_adjacency_matrix)
    loss = criterion(out[data.test_mask], data.ndata['label'][data.test_mask]) + opt.kl * kl
    _, pred = torch.max(out[data.test_mask], dim=1)
    correct = (pred == data.ndata['label'][data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
   # print("test_loss: {:.4f} test_acc: {:.4f}".format(loss.item(), acc))

    return acc


if __name__ == '__main__':
    opt = OptInit().initialize()
    #dataset = Planetoid(root='./dataset/', name='Citeseer')
    # dataset = Planetoid(root='./dataset/', name='Cora')
    # dataset = Planetoid(root='./cora/', name='Cora', split='random',
    #                          num_train_per_class=232, num_val=542, num_test=542)
    # dataset = Planetoid(root='./citeseer',name='Citeseer')
    # dataset = Planetoid(root='./pubmed/', name='Pubmed')
    GRAPH_DICT = {
        "cora": CoraGraphDataset,
        "citeseer": CiteseerGraphDataset,
        "pubmed": PubmedGraphDataset,
        "coauther_cs": CoauthorCSDataset,
        'amazon_photo': AmazonCoBuyPhotoDataset,
        'coauther_phy': CoauthorPhysicsDataset,
        'chameleon': ChameleonDataset,
        'squirrel': SquirrelDataset,
        'actor': ActorDataset,
        'wiki': WikiCSDataset,
        'amac': AmazonCoBuyComputerDataset
    }
    datasetLS = ["cora", "citeseer", 'amazon_photo', 'wiki', "amac", "coauther_cs", "pubmed", 'coauther_phy']

    for name in range(6, 7):
        opt.name = datasetLS[name]

        dataset = GRAPH_DICT[opt.name]()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = dataset[0]

        edges = data.edges()  # 提取边（源节点、目标节点）
        num_nodes = data.num_nodes()  # 节点数量
        adj = sp.coo_matrix((np.ones(edges[0].shape[0]), (edges[0].numpy(), edges[1].numpy())),
                            shape=(num_nodes, num_nodes))  #邻接矩阵

        normalized_adjacency_matrix = get_norm_matrix(adj).to(device)
        data = data.to(device)

        num_node_features = data.ndata['feat'].shape[1]
        num_classes = len(torch.unique(data.ndata['label']))

        filename = os.path.join('./dataset/split_datasets/', opt.name.lower() + '_splits.npy')
        #得到数据划分
        splits_list = np.load(filename)

        print(dataset)
        tem_x = data.ndata['feat'].clone()
        max_mean_acc = -float('inf')

        repeat_times = 1
        results = {}
        for repeat in range(1, repeat_times+1):  # Repeat 5 times
            acc = []  # Store accuracies for each split
            for i in range(10):
                split = splits_list[i]
                train_idx = torch.tensor(np.where(split == 0, True, False)).to(device)
                test_idx = torch.tensor(np.where(split == 1, True, False)).to(device)
                val_idx = torch.tensor(np.where(split == 2, True, False)).to(device)

                data.ndata['feat'][test_idx]= 0
                data.ndata['feat'][val_idx] = 0

                data.train_mask =  train_idx
                data.test_mask = test_idx
                data.val_mask = val_idx

                model = FusionGCN(num_nodes, num_node_features, num_classes, opt)
                #print(device)
                model.to(device)


                criterion = nn.NLLLoss().to(device)
                optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)



                record_dict = {'best_epoch':-1, 'best_val_acc':0, 'best_test_acc':0}

                train()

                print('\n', i)
                acc.append(record_dict['best_test_acc'])
                print(f"{opt.name} now_mean_acc:{sum(acc) * 100/(i+1)}%\n\n", acc, '\n')

                data.ndata['feat'] = tem_x.clone()


            mean_acc = sum(acc) * 10

            results[repeat] = [mean_acc, acc]

            # Update max_mean_acc if necessary
            if mean_acc > max_mean_acc:
                max_mean_acc = mean_acc

        # Add max_mean_acc to results dictionary
        results['max_mean_acc'] = [max_mean_acc]

        # Save results to Excel
        filename = f"exp_mf\\{opt.name}\\exp_mv_repeat{repeat_times}_{max_mean_acc}.xlsx"

        # 获取文件夹路径
        folder_path = os.path.dirname(filename)

        # 如果文件夹不存在，则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存结果到 Excel 文件
        save_results_to_excel(results, filename)
