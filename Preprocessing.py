import anndata
import scanpy as sc
import pandas as pd
def read_data(file_path, file_type):
    adata = []
    if file_type == 'csv':
        adata = anndata.read_csv(file_path)
    if file_type == 'normal':
        df_data = pd.read_csv(file_path)
        data = df_data.to_numpy().T
        ori_metrix = data[1:, 2:].astype(float)
        adata = anndata.AnnData(ori_metrix)
    return adata


def process_data(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.obs['scale_factor'] = adata.obs.n_counts / adata.obs.n_counts.mean()
    sc.pp.log1p(adata)
    adata_hvg = adata.copy()
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=1000, inplace=True, flavor='seurat')
    adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable'].values]
    X_hvg = adata_hvg.X
    return adata_hvg, X_hvg






