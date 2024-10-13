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

def preprocess_raw_data1(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.obs['scale_factor'] = adata.obs.n_counts / adata.obs.n_counts.mean()
    sc.pp.log1p(adata)
    return adata


def process_data(adata):
    adata_hvg = adata.copy()
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=1000, inplace=True, flavor='seurat')
    adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable'].values]
    X_hvg = adata_hvg.X
    return adata_hvg, X_hvg

def data_preprocess(adata, filter_min_counts=True, scale_factor=True, normalize_input=True, logtrans_input=True,
                    counts_per_cell=False, select_gene_desc=True, select_gene_adclust=False, use_count=False):
    if filter_min_counts:
        if use_count:
            sc.pp.filter_genes(adata, min_counts=1)
            sc.pp.filter_cells(adata, min_counts=1)
        else:
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.filter_cells(adata, min_genes=200)

    if scale_factor or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if scale_factor:
        sc.pp.normalize_per_cell(adata)
        adata.obs['scale_factor'] = adata.obs.n_counts / adata.obs.n_counts.median()
    else:
        adata.obs['scale_factor'] = 1.0

    if counts_per_cell:
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)

    if logtrans_input:
        sc.pp.log1p(adata)

    if select_gene_desc:
        sc.pp.highly_variable_genes(adata, n_top_genes=1000, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)
        adata = adata[:, adata.var['highly_variable']]

    if select_gene_adclust:
        sc.pp.highly_variable_genes(adata, min_mean=None, max_mean=None, min_disp=None, n_top_genes=1000)
        adata = adata[:, adata.var.highly_variable]

    if normalize_input:
        sc.pp.scale(adata)

    return adata







