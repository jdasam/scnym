import anndata
from scnym.sctransform import SCTransform
from time import time


path = "/home/svcapp/tbrain_x/azimuth/raw/SeuratProject.h5ad"
adata = anndata.read_h5ad(path)

times = []
times.append(time())
test = SCTransform(adata[:10000], gmean_eps=1-1e-5, n_cells=None, min_cells=500, inplace=False)
times.append(time())
test2 = SCTransform(adata[:3000], gmean_eps=1-1e-5, n_cells=None, min_cells=500, inplace=False)
times.append(time())
test3 = SCTransform(adata[:1000], gmean_eps=1-1e-5, n_cells=None, min_cells=500, inplace=False)
times.append(time())
test4 = SCTransform(adata[:300], gmean_eps=1-1e-5, n_cells=None, min_cells=500, inplace=False)
times.append(time())

print([times[i]-times[i-1] for i in range(1, len(times))])

# SCTransform(adata[:500], gmean_eps=1-1e-5, n_cells=None, min_cells=5000)
# print(adata[:500])