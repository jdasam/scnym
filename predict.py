import scnym
import anndata
import scanpy as sc
import numpy as np
import pickle
from scnym.predict import Predicter
import scnym.utils as utils
import scipy 
scnym_api = scnym.api.scnym_api


if __name__ == "__main__":
    # parser = scnym.scnym_ad.make_parser()
    # args = parser.parse_args()

    with open("new_data/scnym_train_results.pkl",'rb') as f:
        results = pickle.load(f)
    
    geninus = anndata.read_h5ad("/home/svcapp/tbrain_x/SKT_data_corrected/IO_response/geninus_raw_final.h5ad")
    geninus.X = scipy.sparse.csr_matrix(geninus.X, dtype=np.float32)
    # gene_names = geninus.var.index.tolist()
    gene_idx = np.zeros(len(results['gene_names']), dtype=np.int32)
    for i, gene in enumerate(results['gene_names']):
        gene_idx[i] = np.where(geninus.var.index == results['gene_names'][i])[0]
    sc.pp.normalize_per_cell(geninus, counts_per_cell_after=2e3)

    predicter = Predicter(results['model_path'])
    prediction, _ = predicter.predict(utils.get_adata_asarray(geninus[:, gene_idx]))

    prediction_label = results['class_names'][prediction]

    geninus.obs['celltype_pred_scnym'] = prediction_label
    
    geninus.obs.to_csv("geninus_prediction_scnym_ssl.csv")
    geninus.write("geninus_raw_scnym_added.h5ad")
