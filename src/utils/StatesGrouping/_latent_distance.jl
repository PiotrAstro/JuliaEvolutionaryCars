# -------------------------------------------------------------
# latent distance tree builder
# -------------------------------------------------------------

function get_latent_distance_tree(
    encoded_exemplars::Matrix{Float32};
    distance_metric::Symbol=:cosine,
    hclust_latent::Symbol = :ward,  # :single, :complete, :average, :ward
) :: TreeNode
    if distance_metric == :cosine
        distance_premetric = Distances.CosineDist()
    elseif distance_metric == :euclidean
        distance_premetric = Distances.Euclidean()
    elseif distance_metric == :cityblock
        distance_premetric = Distances.Cityblock()
    elseif distance_metric == :mul
        distance_premetric = MulDistance()
    elseif distance_metric == :mul_weighted
        distance_premetric = MulWeightedDistance()
    elseif distance_metric == :mul_norm
        distance_premetric = MulNormDistance()
    else
        throw(ArgumentError("Unknown distance metric: $distance_metric"))
    end
    
    distances = Distances.pairwise(distance_premetric, encoded_exemplars, encoded_exemplars)
    merges = Clustering.hclust(distances, linkage=hclust_latent).merges
    root = _create_tree_hclust(merges)

    return root
end
