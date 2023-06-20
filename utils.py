import copy

import pandas as pd
import numpy as np
import spacy
from sklearn.datasets import fetch_openml
import ipdb
from torch.utils.data import TensorDataset
import torch
import os
import scipy.sparse as sp
import random
from gensim.parsing.preprocessing import remove_stopwords
import nltk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils import data


def sens_to_int(sens_attr, graph=False):
    sens = sens_attr.values
    count = 0
    dic_label ={}
    if graph:
        sens[sens > 0] = 1
    for i in sens:
        if i not in dic_label:
            dic_label[i] = count
            count += 1
    sen_list = []
    for sen in sens:
        a = np.zeros(count)
        a[dic_label[sen]] = 1
        sen_list.append(a.tolist())

    return np.array(sen_list)

def cal_correlation(X_raw, sens_attr, related_attr):
    X_src = pd.get_dummies(X_raw[sens_attr])
    X_relate = pd.get_dummies(X_raw[related_attr])
    print(X_src.shape, X_relate.shape)
    correffics = []
    for i in range(len(X_src.keys())):
        for j in range(len(X_relate.keys())):
            correffic = abs(X_src[X_src.keys()[i]].corr(X_relate[X_relate.keys()[j]]))
            correffics.append(correffic)

    print(len(X_relate.keys()), len(correffics))
    return sum(correffics)/len(correffics)
    # return sum(correffics)


def load_text_data(args, device, max_seq=200):

    # data = pd.read_csv('./data/anime/anime_filtered.csv')
    data = pd.read_csv('./data/anime2/anime_filtered.csv')
    labels = data['ranked'].values
    label_idx = np.where(labels>=0)[0]
    n_classes = labels.max().item() + 1
    labels = np.eye(n_classes)[labels]
    labels = torch.FloatTensor(labels).to(device)
    # sens = data['gender'].values
    sens = data['avg_fav_sco'].values
    sens = np.eye(2)[sens]
    # feat_attr = ['birthday', 'score', 'overall', 'story', 'animation', 'sound'
    #          , 'character', 'enjoyment', 'avg_fav_epi', 'avg_fav_pop', 'avg_fav_sco',
    #           'avg_rev_epi', 'avg_rev_pop', 'avg_rev_ran', 'avg_rev_sco']
    feat_attr = ['gender', 'score', 'overall', 'story', 'animation', 'sound'
        , 'character', 'enjoyment', 'avg_fav_epi', 'avg_fav_pop', 'avg_fav_sco',
                 'avg_rev_epi', 'avg_rev_pop', 'avg_rev_ran', 'avg_rev_sco']

    features = data[feat_attr].values

    text = data['text'].values

    vocab = np.genfromtxt('./data/anime2/vocab', dtype=np.str)
    embedding = np.genfromtxt('./data/anime2/vocab_emb', dtype=np.float64)
    # embedding = None
    count = 1
    dic_vocab = {}

    for i in vocab:
        dic_vocab[i] = count
        count += 1
    vocab_len = len(dic_vocab)
    text_data = []
    for k in text:
        seq = []
        for word in k.split(" "):
            if word in dic_vocab:
                seq.append(dic_vocab[word])
        while len(seq)<max_seq:
            seq.append(0)
        text_data.append(seq[0:max_seq])

    text_data = torch.LongTensor(text_data).to(device)
    embedding = torch.FloatTensor(embedding).to(device)
    random.shuffle(label_idx)
    # idx_train = np.genfromtxt("./data/anime2/idx_train", dtype=int)
    # idx_train = label_idx
    # random.shuffle(idx_train)
    # idx_train = idx_train[0:int(0.5 * len(idx_train))].tolist() + label_idx[:int(0.1 * len(label_idx))].tolist()
    # idx_train = np.array(idx_train)
    idx_train = label_idx[:int(0.5 * len(label_idx))]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    idx_test = label_idx[int(0.75 * len(label_idx)):]
    print(features.shape)

    return features, text_data, labels, sens, n_classes, idx_train, idx_val, idx_test, vocab_len, embedding

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def load_graph_data(args, device):
    if args.dataset == 'pokec_z':
        dataset = 'region_job'
    else:
        dataset = 'region_job_2'
    sens_attr = 'gender'
    predict_attr = "I_am_working_in_field"
    path = "./data/pokec/"
    if args.dataset == 'bail':
        dataset = 'bail'
        sens_attr = 'WHITE'
        predict_attr = "RECID"
        path = "./data/bail/"

    if args.dataset == 'credit':
        dataset = 'credit'
        sens_attr = 'Age'
        predict_attr = "Married"
        path = "./data/credit/"
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))
    graph_embedding = np.genfromtxt(
        os.path.join(path, "{}.embedding".format(dataset)),
        skip_header=1,
        dtype=float
    )
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))

    embedding_df = pd.DataFrame(graph_embedding)
    # print(embedding_df[0])
    embedding_df[0] = embedding_df[0].astype('category')
    embedding_df[0].cat.set_categories(list(idx_features_labels["user_id"]), inplace=True, ordered=True)
    embedding_df = embedding_df.sort_values(0, ascending=True)
    embedding_df = embedding_df.rename(index=int, columns={0: "user_id"})
    # print(embedding_df)
    header = list(embedding_df.columns)
    header.remove("user_id")

    embedding = sp.csr_matrix(embedding_df[header], dtype=np.float32)
    # print(embedding_df['user_id'])
    # print(idx_features_labels['user_id'])
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove("Single")

    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels > 1] = 1
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # index_all = np.arange(0,embedding.shape[0])
    # index_all_left = np.repeat(index_all, embedding.shape[0])
    # index_all_right = np.tile(index_all, embedding.shape[0])
    # edges_all = np.vstack(index_all_left, index_all_right)
    # pos_edge = np.vstack((edges[:, 0], edges[:, 1]))
    neg_edge = np.genfromtxt(os.path.join(path, "{}_relationship_neg.txt".format(dataset)), dtype=int)
    pos_edge = np.genfromtxt(os.path.join(path, "{}_relationship_train.txt".format(dataset)), dtype=int)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
    # print(adj.todense().shape)

    # embedding = np.sum(adj.todense(), axis=1)
    adj_num = adj.todense()
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to(device)
    features = np.array(features.todense())

    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:int(0.5 * len(label_idx))]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    idx_test = label_idx[int(0.75 * len(label_idx)):]

    if dataset == 'bail':
        label_num = 2000
    if dataset == 'credit':
        label_num = 2000
    if dataset =='bail' or dataset=='credit':
        idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_num)]
        # idx_val = label_idx[len(idx_train):len(idx_train)+int((len(label_idx)-len(idx_train))*0.25)]
        # idx_test = label_idx[len(idx_train)+int((len(label_idx)-len(idx_train))*0.25):]


    feat = copy.deepcopy(features)
    # scaler = StandardScaler().fit(features[idx_train])
    # features[idx_train] = scaler.transform(features[idx_train])
    # features[idx_val] = scaler.transform(features[idx_val])
    # features[idx_test] = scaler.transform(features[idx_test])


    embedding = np.array(embedding.todense())
    # scaler = StandardScaler().fit(embedding[idx_train])
    # embedding[idx_train] = scaler.transform(embedding[idx_train])
    # embedding[idx_val] = scaler.transform(embedding[idx_val])
    # embedding[idx_test] = scaler.transform(embedding[idx_test])

    # labels = labels
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    n_classes = labels.max().item()+1


    sens = sens_to_int(idx_features_labels[sens_attr], graph=True)
    features = torch.FloatTensor(features).to(device)
    if dataset =='bail' or dataset=='credit':
        features = feature_norm(features)

    feat = torch.FloatTensor(feat).to(device)
    labels = np.eye(n_classes)[labels]
    labels = torch.FloatTensor(labels).to(device)
    # sens = torch.IntTensor(sens).to(device)
    # print(embedding)
    embedding = torch.FloatTensor(embedding).to(device)
    # embedding = feature_norm(embedding)
    # print(features.shape, embedding.shape)
    # random.shuffle(sens_idx)
    # embedding = features

    return adj, features, labels, sens, n_classes, embedding, idx_train, idx_val, idx_test, pos_edge, neg_edge, feat, adj_num

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(args):
    if args.dataset == 'adult':
        data = fetch_openml(data_id=1590)

        header = list(data.data.columns)
        print(header)
        if args.method == 'remove':
            for attr in args.related:
                header.remove(attr)

        related_data = data.data[args.related]
        data_frame = data.data
        sensitive_attr = data_frame[args.s]
        header.remove(args.s)
        data.data = data.data[header]

        X = pd.get_dummies(data.data)
        X = X.sort_index(axis=1)
        X_related = pd.get_dummies(related_data)
        X_related = X_related.sort_index(axis=1)

        y_true = ((data.target == '>50K') * 1).values
        n_classes = y_true.max() + 1

        y_true = np.eye(n_classes)[y_true]

        # data_frame = pd.DataFrame(data.data, columns=data.feature_names)
        # sensitive attribute obtain


        # print(sensitive_attr.value_counts())#1 for male, 0 for female
        # print(data.data.groupby('sensitive_attr')['relationship'].value_counts()/data.data.groupby('sensitive_attr')['relationship'].count())
        data = fetch_openml(data_id=1590)
        for relate in args.related:
            coef = cal_correlation(data.data, args.s, relate)
            print('coefficient between {} and {} is: {}'.format(args.s, relate, coef))
            data.data['target'] = data.target
            coef = cal_correlation(data.data, 'target', relate)
            print('coefficient between {} and {} is: {}'.format('target', relate, coef))


    elif args.dataset == 'law':
        args.s = "sex"
        predict_attr = "admit"

        data_name = 'processed_data'
        idx_features_labels = pd.read_csv("./data/law_school/{}.csv".format(data_name))
        idx_features_labels = idx_features_labels.sort_index(axis=1)

        print(idx_features_labels.keys())

        header = list(idx_features_labels.columns)

        sensitive_attr = idx_features_labels[args.s]
        data_frame = idx_features_labels[header]

        header.remove(predict_attr)
        related_data = idx_features_labels[args.related]
        X_related = pd.get_dummies(related_data)
        X_related = X_related.sort_index(axis=1)

        if args.method == 'remove':
            for attr in args.related:
                header.remove(attr)

        # X = np.array(idx_features_labels[header], dtype=np.float32)
        X = idx_features_labels[header]
        # y_true = idx_features_labels[predict_attr].values
        y_true = idx_features_labels[predict_attr].values
        label_idx = np.where(y_true >= 0)[0]
        X = X.iloc[label_idx, :]
        y_true = y_true[label_idx]
        n_classes = y_true.max() + 1
        y_true = np.eye(n_classes)[y_true]

        # data_frame = data_frame.iloc[label_idx, :]
        # sensitive_attr = sensitive_attr.iloc[label_idx]

        X = pd.get_dummies(X)

        # for relate in args.related:
        #     coef = cal_correlation(data_frame, args.s, relate)
        #     print('coefficient between {} and {} is: {}'.format(args.s, relate, coef))

        # print(data_frame.groupby(args.s)[predict_attr].value_counts()/data_frame.groupby(args.s)[predict_attr].count())

    elif args.dataset == 'compas':
        args.s = 'race'
        predict_attr = "is_recid"

        data_name = 'Processed_Compas'
        idx_features_labels = pd.read_csv("./data/{}.csv".format(data_name))

        print(idx_features_labels.keys())

        header = list(idx_features_labels.columns)

        related_data = idx_features_labels[args.related]
        X_related = pd.get_dummies(related_data)
        X_related = X_related.sort_index(axis=1)

        sensitive_attr = idx_features_labels[args.s]
        data_frame = idx_features_labels[header]

        header.remove(predict_attr)

        if args.method == 'remove':
            for attr in args.related:
                header.remove(attr)

        # X = np.array(idx_features_labels[header], dtype=np.float32)
        X = idx_features_labels[header]
        # y_true = idx_features_labels[predict_attr].values
        y_true = idx_features_labels[predict_attr].values


        n_classes = y_true.max() + 1

        y_true = np.eye(n_classes)[y_true]

        # for relate in args.related:
        #     coef = cal_correlation(data_frame, args.s, relate)
        #     print('coefficient between {} and {} is: {}'.format(args.s, relate, coef))
        #
        #     coef = cal_correlation(data_frame, predict_attr, relate)
        #     print('coefficient between {} and {} is: {}'.format(predict_attr, relate, coef))

        # ipdb.set_trace()
        # print(data_frame.groupby(args.s)[predict_attr].value_counts()/data_frame.groupby(args.s)[predict_attr].count())

        X = pd.get_dummies(X)
        X = X.sort_index(axis=1)
    elif args.dataset == 'movielens':
        args.s = 'gender'
        predict_attr = "rating"
        data_name = 'movielens'
        feat_name = ['age', 'occupation']
        idx_features_labels = pd.read_csv("./data/movielens/{}.data".format(data_name))
        header = list(idx_features_labels.columns)
        X = idx_features_labels[feat_name]
        header.remove(predict_attr)
        header.remove(args.s)
        for name in feat_name:
             header.remove(name)
        sensitive_attr = idx_features_labels[args.s]
        related_data = idx_features_labels[header]
        X_related = pd.get_dummies(related_data)
        X_related = X_related.sort_index(axis=1)

        y_true = idx_features_labels[predict_attr].values

        n_classes = y_true.max() + 1

        y_true = np.eye(n_classes)[y_true]

        X = pd.get_dummies(X)
        X = X.sort_index(axis=1)
    return X, X_related,y_true, sensitive_attr, n_classes

class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes, device, graph=False):
        if graph:
            tensors = (torch.from_numpy(df).long().to(device) for df in dataframes)
        else:
            tensors = (self._df_to_tensor(df) for df in dataframes)
        self.device = device
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, np.ndarray):
            return torch.from_numpy(df).float().to(self.device)
        return torch.from_numpy(df.values).float().to(self.device)

class Dataload(data.Dataset):

    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node
    def __getitem__(self, index):
        return index
        # adj_batch = self.Adj[index]
        # adj_mat = adj_batch[index]
        # b_mat = torch.ones_like(adj_batch)
        # b_mat[adj_batch != 0] = self.Beta
        # return adj_batch, adj_mat, b_mat
    def __len__(self):
        return self.Node
