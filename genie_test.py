import numpy as np
import genieclust as gc

# Generate some data
np.random.seed(0)
X = np.random.randn(10_000, 32)
clust = gc.Genie(compute_full_tree=True, verbose=True)
clust.fit(X)
print("Done!")
