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

def trajectory_analysis(embedding, cluster, output_folder):
    data = np.load(embedding)
    clusters = np.load(cluster)
    adata = anndata.AnnData(X=data)

    cell_types = {
        0: 'OPC',
        1: 'astrocytes',
        2: 'endothelial',
        3: 'fetal_quiescent',
        4: 'fetal_replicating',
        5: 'microglia',
        6: 'neurons',
        7: 'oligodendrocytes'
    }#Darmanis

    adata.obs['cell_type'] = [cell_types[x] for x in clusters]

    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)

    # UMAP
    sc.pl.umap(adata, color='cell_type', title='Cluster Visualization', legend_loc='on data', legend_fontsize=14)
    # PAGA
    sc.tl.paga(adata, groups='cell_type')

    plt.rcParams.update({'font.size': 20})
    empty_labels = [''] * len(cell_types)
    sc.pl.paga_compare(adata, basis='umap', color='cell_type', title='Darmanis', labels=empty_labels,
                       legend_loc='False', legend_fontsize=15, show=False)
    fig = plt.gcf()
    axes = fig.get_axes()

    for ax in axes:
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                 linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    colors = sc.pl.palettes.default_20[:len(cell_types)]
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=cell_type)
        for i, cell_type in enumerate(cell_types.values())
    ]
    plt.subplots_adjust(bottom=0.4)

    fig.legend(handles=legend_elements, loc='lower center', ncol=len(cell_types),
               fontsize=14, handletextpad=0.1, columnspacing=0.3, bbox_to_anchor=(0.5, -0.02),
               bbox_transform=plt.gcf().transFigure, frameon=False)

    fig.savefig(output_folder, format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    adata.write(os.path.join(output_folder, 'adata.csv'))
    print(f"Results and visualizations are saved in {output_folder}")

def Marker_Gene_Analysis(embedding, cluster, gene, output_folder):

    if not (os.path.exists(embedding) and os.path.exists(cluster) and os.path.exists(gene)):
        print("One or more input files do not exist. Please check the file paths.")
        return

    gene_mapping_df = pd.read_csv(gene, header=None)
    embedding = pd.read_csv(embedding)
    clusters = np.load(cluster)

    cluster_mapping = {
        0: 'MHC class II', 1: 'PSC', 2: 'acinar', 3: 'alpha',
        4: 'beta', 5: 'co-expressioon', 6: 'delta', 7: 'ductal',
        8: 'endathelial', 9: 'epsilon', 10: 'gamma', 11: 'mast', 12: 'unclassified endocrine'
    }#Segerstolpe
    mapped_clusters = np.vectorize(cluster_mapping.get)(clusters)

    adata = sc.AnnData(X=embedding.values)
    adata.obs['clusters'] = mapped_clusters
    adata.var_names = gene_mapping_df[1]

    valid_clusters = [c for c in np.unique(adata.obs['clusters']) if sum(adata.obs['clusters'] == c) > 1]

    if len(valid_clusters) > 1:
        sc.tl.rank_genes_groups(adata, groupby='clusters', groups=valid_clusters, method='wilcoxon')
    else:
        print("Not enough valid clusters for differential expression analysis.")
        return

    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    marker_genes = {group: adata.uns['rank_genes_groups']['names'][group][:25] for group in groups}
    top2_marker_genes = {group: genes[:2] for group, genes in marker_genes.items()}

    cluster_to_id = {name: idx for idx, name in enumerate(adata.obs['clusters'].unique())}
    adata.obs['cluster_ids'] = adata.obs['clusters'].map(cluster_to_id)

    # Plot
    if 'cluster_ids' in adata.obs.columns:
        plt.rcParams.update({'font.size': 17})
        fig, ax = plt.subplots(figsize=(14, 9))
        plt.subplots_adjust(left=0.1, right=0.8, top=0.7, bottom=0.2)
        sc.pl.dotplot(adata, var_names=top2_marker_genes, groupby='cluster_ids', ax=ax, show=False)
        plt.savefig(output_folder, dpi=600, bbox_inches='tight', format='pdf')
        plt.show()
    else:
        print("The clustering results 'clusters' do not exist; please check your input data.")


def visualize_cell_types(embedding, cluster, output_folder):

    data = np.load(embedding)
    adata = anndata.AnnData(X=data)
    clusters = np.load(cluster)

    cell_types = {
        0: 'MHC class II', 1: 'PSCs', 2: 'acinar cells', 3: 'α-cells',
        4: 'β-cells', 5: 'co-expression', 6: 'δ-cells', 7: 'ductal cells',
        8: 'endothelial cells', 9: 'ε-cells', 10: 'γ-cells', 11: 'mast cells', 12: 'unclassified endocrine'
    }# Segerstolpe

    adata.obs['clusters'] = clusters
    adata.obs['cell_type'] = [cell_types[x] for x in clusters]

    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)

    fig, ax = plt.subplots(figsize=(8, 8.5))
    sc.pl.umap(adata, color='cell_type', title='annotation_Segerstolpe', legend_fontsize=16, ax=ax, s=90, show=False)

    fig.subplots_adjust(left=0.03, right=0.86, top=0.85, bottom=0.1)
    ax.set_title('annotation_Segerstolpe', fontname='Times New Roman', fontsize=40, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, scatterpoints=1, fontsize=25, markerscale=2,
                    bbox_to_anchor=(0.98, 0.5), loc='center left', frameon=False)

    plt.savefig(output_folder, format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

