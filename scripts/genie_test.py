import numpy as np
import genieclust as gc

# Generate some data
np.random.seed(0)
X = np.random.randn(1_000, 32)
clust = gc.Genie(n_clusters = 10, compute_full_tree=True, compute_all_cuts=True, verbose=True)
clust.fit(X)

print("Done!")
