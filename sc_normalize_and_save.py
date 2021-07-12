import anndata
import scanpy as sc
import numpy as np
import scipy

path = "/home/svcapp/tbrain_x/azimuth/pbmc_multimodal_raw.h5ad"
unlabeled_path = "/home/svcapp/tbrain_x/SKT_data_corrected/IO_response/geninus_raw_final.h5ad"
labeled_path = "/home/svcapp/tbrain_x/azimuth/pbmc_multimodal.h5ad"

adata = anndata.read_h5ad(path)
adata2 = anndata.read_h5ad(labeled_path)

adata.obs = adata2.obs
del adata2

unlabeled_adata = anndata.read_h5ad(unlabeled_path)
unlabeled_adata.X = scipy.sparse.csr_matrix(unlabeled_adata.X, dtype=np.float32)
sc.pp.normalize_per_cell(unlabeled_adata, counts_per_cell_after=2e3)

# adata = anndata.read_h5ad(path)
sc.pp.normalize_per_cell(adata, counts_per_cell_after=2e3)
sc.pp.filter_genes(adata, min_cells=20000)
sc.pp.filter_cells(adata, min_genes=100)


adata.obs["domain"] = "azimuth"
gene_names = adata.var_names.tolist()
gene_idx = np.zeros(len(gene_names), dtype=np.int32)
for i, gene in enumerate(gene_names):
    gene_idx[i] = np.where(unlabeled_adata.var.index == gene_names[i])[0]
unlabeled_adata = unlabeled_adata[:, gene_idx]
unlabeled_adata.obs["celltype.l2"] = "Unlabeled"
unlabeled_adata.obs["domain"] = "geninus"
sc.pp.filter_cells(unlabeled_adata, min_genes=100)

num_unlabel = unlabeled_adata.shape[0]
end_idx = num_unlabel // 5

for i in range(4):
    print(i)
    adata = anndata.concat([adata, unlabeled_adata[:end_idx]])
    print(f"{i}th concat finished")
    unlabeled_adata = unlabeled_adata[end_idx:]
print("try concat")
adata = anndata.concat([adata, unlabeled_adata])
print("concat end")
del unlabeled_adata
print("del end")

adata.write('pbmc_multimodal_raw_labeled.h5ad')
