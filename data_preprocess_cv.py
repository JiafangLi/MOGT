import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import torch as t
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split, StratifiedKFold,StratifiedShuffleSplit
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from config import config_load
from sklearn import preprocessing

import random
from collections import Counter

EPS = 1e-8

def arg_parse():
    parser = argparse.ArgumentParser(description="Data Preprocess.")
    parser.add_argument('-j', '--joint', dest='joint', help="to package joint data", action="store_true")
    parser.add_argument('-r', '--reverse', dest='reverse', help="to package reverse patient data", action="store_true")
    return parser.parse_args()




def split_data(CV_FOLDS,labeled_idx,labeled_lab):
    RANDOM_SEED = 42
    train_idx_list, valid_idx_list = [], []
    train_valid_idx, test_idx, train_valid_lab, test_lab = train_test_split(
        labeled_idx, labeled_lab, test_size=0.2, stratify=labeled_lab, random_state=RANDOM_SEED)

    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for train_labeled_idx, valid_labeled_idx in skf.split(train_valid_idx, train_valid_lab):
        train = [train_valid_lab[i] for i in train_labeled_idx]
        valid = [train_valid_lab[i] for i in valid_labeled_idx]
        valid_idx_list.append([train_valid_idx[i] for i in valid_labeled_idx])
        train_idx_list.append([train_valid_idx[i] for i in train_labeled_idx]) 
    
    return train_idx_list,valid_idx_list,test_idx

def split_core_data(CV_FOLDS,labeled_idx,labeled_lab):
    RANDOM_SEED = 42
    train_idx_list, valid_idx_list,test_idx_list = [], [],[]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for train_labeled_idx, test_labeled_idx in skf.split(labeled_idx, labeled_lab):
        X = [labeled_idx[i] for i in train_labeled_idx]
        y = [labeled_lab[i] for i in train_labeled_idx]
        train_index,valid_index,train_lab,valid_lab = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
        test_idx_list.append([labeled_idx[i] for i in test_labeled_idx])
        train_idx_list.append(train_index)
        valid_idx_list.append(valid_index)

    
    return train_idx_list,valid_idx_list,test_idx_list



def scale_data(data,fold):
    X_train = torch.nonzero(~data[0].train_mask[:,fold]).squeeze()
    X_valid = torch.nonzero(~data[0].valid_mask[:,fold]).squeeze()
    X_test = torch.nonzero(~data[0].test_mask).squeeze()
    scaler = preprocessing.StandardScaler().fit(data[0].x[X_train][:,1:])

    # scale data
    data[0].x[X_train][:,1:] = torch.from_numpy(scaler.transform(data[0].x[X_train][:,1:]))
    data[0].x[X_valid][:,1:] = torch.from_numpy(scaler.transform(data[0].x[X_valid][:,1:]))
    data[0].x[X_test][:,1:] = torch.from_numpy(scaler.transform(data[0].x[X_test][:,1:]))

    return data


def read_table_to_np(table_file, sep='\t', dtype=float, start_col=1):
    data = pd.read_csv(table_file, sep=sep)
    data = data.iloc[:, start_col:].to_numpy().astype(dtype)
    return data


def get_snp_mat(disease="SCZ"):
    data_dir = "data/"+ f"/{disease}/{disease}Network.csv"
    data = pd.read_csv(data_dir,sep=",").to_numpy()[:, :]
    print(f"Loading SNP matrix from {data_dir} ......")
    return data



def get_label(disease="SCZ", reverse=False, slice=False):
    """
    Read label data, where some nodes have labels and others do not. For the nodes with labels, change the labels from -1 to 0.

    Returns:
    labels:         ndarray(num_nodes, 2). First col is 1 for negative nodes, and second col is 1 for positive nodes.
    labeled_idx:    list(num_nodes). Indices of labeled nodes.
    """
    label_dir = "data/"+ f"/{disease}/{disease}.csv"
    data = pd.read_csv(label_dir)
    data = data.loc[:,["gene_name","gene_id","label"]]
    data = data.iloc[:, 2:].to_numpy().astype(int).transpose()[0]
    if slice: data = data[slice]

    labeled_idx = []
    labels = np.zeros((len(data), 2), dtype=float)
    for i in range(len(data)):
        if data[i] !=-1:
            labeled_idx.append(i)
            if data[i] == 1:
                labels[i][1] = 1
            else:
                labels[i][0] = 1
    return labels, labeled_idx

def get_node_feat(disease = "SCZ"):
    data_dir = "data/" + disease + "/" + disease+".csv"
    feat = pd.read_csv(data_dir,sep=",")
    feat = feat.iloc[:, 3:]
    feat = feat.to_numpy().astype(float)
    pos = np.arange(feat.shape[0])
    print(f"data.dim:{feat.shape}")
    return feat, pos

def get_edge_threshold(mat,edge_threshold):
    flat_arr = mat.flatten()
    top_percent = int(len(flat_arr) * edge_threshold * 0.01)
    sort_arr = np.sort(flat_arr)
    threshold = sort_arr[-top_percent]
    return threshold

def get_random_edge(mat,edge_threshold):
    num_nodes = int(mat.shape[0])
    select_num = int(num_nodes * np.sqrt(edge_threshold) * 0.1) 
    print(f"selct_num:{select_num}")
    x = random.sample(range(num_nodes), select_num)
    y = random.sample(range(num_nodes), select_num)
    return x,y

def construct_edge(mat,edge_threshold,random = True):
    """
    Construct edges from adjacent matrix.

    Parameters:
    ----------
    mat:    ndarray(num_nodes, num_nodes).
                PPI matrix from get_ppi_mat().

    Returns:
    edges:      list(num_edges, 2). 
    edge_dim:   int.
                Dim of edge features.
    val:        list(num_edges, ).
                Edge features(=[1] * num_edges in current version).
    """
    num_nodes = mat.shape[0]
    edges = []
    val = []
    threshold = 0
    if random:
        x,y = get_random_edge(mat,edge_threshold)
    else:
        threshold = get_edge_threshold(mat,edge_threshold)
        x,y =  range(num_nodes),range(num_nodes)
    for i in x:
        for j in y:
            if mat[i,j] > threshold:
                edges.append([i, j])
                val.append(mat[i, j])

    edge_dim = 1
    edges = np.transpose(edges)
    val = np.reshape(val, (-1, edge_dim))

    return edges, edge_dim, val


def build_pyg_data(gene_list,node_mat, node_lab, mat, pos,edge_threshhold,random):
    x = t.tensor(node_mat, dtype=torch.float)
    y = t.tensor(node_lab, dtype=torch.long) 
    gene = gene_list
    pos = t.tensor(pos, dtype=torch.int)
    edge_index, edge_dim, edge_feat = construct_edge(mat,edge_threshhold,random)
    edge_index = t.tensor(edge_index, dtype=torch.long)
    edge_feat = t.tensor(edge_feat, dtype=torch.float)
    data = Data(x=x.clone(), y=y.clone(), edge_index=edge_index,
                edge_attr=edge_feat, pos=pos, edge_dim=edge_dim,
                gene_name = gene)
    print(
        f"Number of edges: {data.num_edges}, Dimensionality of edge: {edge_dim},\nNubmer of nodes: {data.num_nodes}")

    return data


class DiseaseDataset(InMemoryDataset):
    def __init__(self, data=None):
        super(DiseaseDataset, self).__init__('.', None, None)
        self.data = data or get_data()
        self.data.num_classes = 2

        self.data, self.slices = self.collate([self.data])

    def get_idx_split(self, i):
        train_idx = torch.where(self.data.train_mask[:, i])[0]
        test_idx = torch.where(self.data.test_mask[:, i])[0]
        valid_idx = torch.where(self.data.valid_mask[:, i])[0]

        return {
            'train': train_idx,
            'test': test_idx,
            'valid': valid_idx
        }

    def get_feature_dict(self):
        if self.data.num_node_features == 16:
            feat_cols = ["de","brain_cp","brain_gz","neuron_count","oligo_count","micro_count","adolescence_parietal.lobe",
                        "adolescence_frontal.lobe","adolescence_temporal.lobe","adolescence_cerebellum",
                        "adolescence_occipital.lobe","adulthood_parietal.lobe","adulthood_frontal.lobe","adulthood_temporal.lobe",
                        "adulthood_cerebellum","adulthood_occipital.lobe"]
            feature_idx = {value: index for index, value in enumerate(feat_cols)}
        elif self.data.num_node_features == 15:
            feat_cols = ["brain_cp","brain_gz","neuron_count","oligo_count","micro_count","adolescence_parietal.lobe",
                        "adolescence_frontal.lobe","adolescence_temporal.lobe","adolescence_cerebellum",
                        "adolescence_occipital.lobe","adulthood_parietal.lobe","adulthood_frontal.lobe","adulthood_temporal.lobe",
                        "adulthood_cerebellum","adulthood_occipital.lobe"]
            feature_idx = {value: index for index, value in enumerate(feat_cols)}

        return feature_idx

from torch.utils.data import Dataset, DataLoader

class NodeDataset(Dataset):
    def __init__(self, data, node_indices):
        self.data = data
        self.node_indices = node_indices

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        node_id = self.node_indices[idx]
        x = self.data.x[node_id]
        y = self.data.y[node_id]
        return x, y 


    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def create_cv_dataset(train_idx_list, valid_idx_list, test_idx, data=None):
    num_nodes = data.num_nodes
    num_folds = len(train_idx_list)

    train_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    valid_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for i in range(num_folds):
        train_mask[train_idx_list[i], i] = True
        valid_mask[valid_idx_list[i], i] = True
    test_mask[test_idx] = True

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    data.unlabeled_mask = ~torch.logical_or(
        data.train_mask[:, 0], torch.logical_or(data.valid_mask[:, 0], data.test_mask))
    cv_dataset = DiseaseDataset(data=data)

    return cv_dataset

def get_data(configs, disease = "SCZ"):

    dataset_dir = "data/"+disease+"/" + disease +"_dataset.pkl"
    print(f"Loading dataset from: {dataset_dir} ......")
    with open(dataset_dir, 'rb') as f:
        cv_dataset = pickle.load(f)

        return cv_dataset


if __name__ == "__main__":
    configs = config_load.get()
    args = arg_parse()
    configs["load_data"] = False

    data = get_data(configs)