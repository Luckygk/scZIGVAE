import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment

def _correlation(data_numpy, k, corr_type='pearson'):
    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    return corr, neighbors

def get_edgelist_PKNN(datasetName,X_hvg,k,type):
    if type == 'PKNN':
        distances, neighbors = _correlation(data_numpy=X_hvg, k=k )
    cutoff = np.mean(np.nonzero(distances), axis=None)
    edgelist = []
    for i in range(
            neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                pair = (str(i), str(neighbors[i][j]))
                distance = distances[i][j]
                if distance < cutoff:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)

    return distances, neighbors, cutoff, edgelist


def load_separate_graph_edgelist(edgelist_path):
    with open(edgelist_path, 'r') as edgelist_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edgelist_file.readlines()]
    return edgelist

def create_graph(edges, X):
    num_nodes = X.shape[0]
    edge_index = np.array(edges).astype(int).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)
    scaler_X = torch.from_numpy(MinMaxScaler().fit_transform(X))
    data_obj = Data(edge_index=edge_index, x=scaler_X)
    return data_obj


def eva(y_true, y_pred, epoch=0):
    res_ari=0.0000
    res_acc = 0.0000
    res_nmi = 0.0000
    acc = cluster_accuracy(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if res_ari < ari:
        res_ari = ari
        res_nmi = nmi
        res_acc = acc
    print(epoch, ':acc {:.4f}'.format(res_acc), ', nmi {:.4f}'.format(res_nmi), ', ari {:.4f}'.format(res_ari))

def cluster_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return acc
