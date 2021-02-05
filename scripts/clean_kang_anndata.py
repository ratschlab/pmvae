import sys
import anndata

_, path = sys.argv

data = anndata.read(path)
data.obs = data.obs[['condition', 'cell_type']]
data.uns = dict()
data.obsm = None
data.varm = None

data.write(path)
