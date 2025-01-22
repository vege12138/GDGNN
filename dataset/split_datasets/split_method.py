import os

import numpy as np
import torch
from collections import namedtuple, Counter

from sklearn.model_selection import train_test_split
from dgl.data import (
    CoraGraphDataset,
    CiteseerGraphDataset,
    PubmedGraphDataset,
    CoauthorCSDataset,
    AmazonCoBuyPhotoDataset,
    CoauthorPhysicsDataset,
    ActorDataset, AmazonCoBuyComputerDataset
)

GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "coauther_cs": CoauthorCSDataset,
    'amazon_photo': AmazonCoBuyPhotoDataset,
    'coauther_phy':CoauthorPhysicsDataset,
    'actor':ActorDataset,
    'amac':AmazonCoBuyComputerDataset,
}

def split_datasets(label, method):
    split_list = []
    num_split = 10
    data_list = [i for i in range(len(label))]

    label_ = np.array(label)
    num_class = len(Counter(label_))
    print("The number of class: ", num_class)
    for random_state in range(num_split):
        class_indices = {}

        for i, l in enumerate(label.tolist()):
            if l not in class_indices.keys():
                class_indices[l] = []
            class_indices[l].append(i)

        train_indices = []
        test_indices = []
        val_indices = []
        for l, indices in class_indices.items():
            if len(indices) == 1:
                train_indices.extend(indices)
            else:
                if method == 'hetero':
                    train_idx, test_val_idx = train_test_split(indices, test_size=0.4, random_state=random_state)
                    test_idx, val_idx = train_test_split(test_val_idx, test_size=0.5, random_state=random_state)
                if method == 'semi':
                    train_idx, x_temp, y_train, y_temp = train_test_split(data_list, label, train_size=20 * num_class,
                                                                          stratify=label, random_state=random_state)
                    val_idx, x_tmp, y_val, y_tmp = train_test_split(x_temp, y_temp, train_size=500, stratify=y_temp,
                                                                    random_state=random_state)
                    test_idx, x_t, y_test, y_t = train_test_split(x_tmp, y_tmp, train_size=1000, stratify=y_tmp,
                                                                  random_state=random_state)
                if method == 'full':
                    train_idx, test_val_idx = train_test_split(indices, test_size=0.8, random_state=random_state)
                    test_idx, val_idx = train_test_split(test_val_idx, test_size=0.75, random_state=random_state)

                train_indices.extend(train_idx)
                test_indices.extend(test_idx)
                val_indices.extend(val_idx)

        split_indices = torch.tensor(np.full(len(label), 3, dtype=int))

        split_indices[train_indices] = 0
        split_indices[test_indices] = 1
        split_indices[val_indices] = 2

        split_libel = split_indices

        # print(split_libel)

        split_list.append(split_libel)
    splits = torch.stack(split_list, dim=0)
    return splits.numpy()

if __name__=='__main__':
    datasetLS = ["cora", "citeseer", 'amazon_photo', 'wiki', "amac", "coauther_cs", "pubmed", 'coauther_phy']
    index = -2
    dataset_name = datasetLS[index]
    dataset = GRAPH_DICT[dataset_name]()
    data = dataset[0]
    label = data.ndata['label']

    method = 'semi' if dataset_name in ["cora", "citeseer", 'amazon_photo'] else 'full'
    splits_list = split_datasets(label, method)

    pre_split_path = 'split'
    if not os.path.exists(pre_split_path):
        os.makedirs(pre_split_path)
    np.save(os.path.join(pre_split_path, dataset_name + '_splits.npy'), splits_list)