
function _get_exemplars(
        encoded_states::Matrix{Float32},
        encoder::NeuralNetwork.AbstractNeuralNetwork,
        n_clusters::Int;
        distance_metric::Symbol=:cosine, # :cosine or :euclidean or cityblock
        exemplars_clustering::Symbol=:genie # :genie or :pam or :kmedoids
    ) :: Tuple{Vector{Int}, TreeNode}
    if distance_metric == :cosine
        distance_premetric = Distances.CosineDist()
    elseif distance_metric == :euclidean
        distance_premetric = Distances.Euclidean()
    elseif distance_metric == :cityblock
        distance_premetric = Distances.Cityblock()
    else
        throw(ArgumentError("Unknown distance metric: $distance_metric"))
    end

    if exemplars_clustering == :genie
        return exemplars_genie(encoded_states, n_clusters, distance_premetric)
    elseif exemplars_clustering == :pam
        return exemplars_pam(encoded_states, n_clusters, distance_premetric)
    elseif exemplars_clustering == :kmedoids
        return exemplars_kmedoids(encoded_states, n_clusters, distance_premetric)
    else
        throw(ArgumentError("Unknown exemplars_clustering: $exemplars_clustering"))
    end

    # normalized_states = GenieClust._normalize_cosine(encoded_states)
    # @time distances = 1 .- (normalized_states' * normalized_states)
    # @time exemplars_pam = Clustering.kmedoids(distances, n_clusters).medoids  # PAM.pam(distances, n_clusters)
    # # @time exemplars_pam, _ = PAM.pam(distances, n_clusters)

    # Plots.scatter(encoded_states[1, :], encoded_states[2, :], label="States", size=(1500, 1500), markerstrokewidth = 0.1)
    # Plots.scatter!(encoded_states[1, exemplars], encoded_states[2, exemplars], label="Exemplars_genie", markerstrokewidth = 0.1)
    # Plots.scatter!(encoded_states[1, exemplars_pam], encoded_states[2, exemplars_pam], label="Exemplars_pam", markerstrokewidth = 0.1, color=:purple)
    # Plots.savefig("log/encoded_states.png")
end

# ------------------------------------------------------------------------------------------
# exemplars functions

function exemplars_genie(
        encoded_states::Matrix{Float32},
        n_clusters::Int,
        distance_premetric::Distances.PreMetric
    ) :: Tuple{Vector{Int}, TreeNode}
    merge_clusters = GenieClust.genie_clust(encoded_states; distance_callable=distance_premetric)
    tree = _create_exemplar_tree_number(merge_clusters, encoded_states, n_clusters, distance_premetric)
    exemplars = tree.elements

    elements = collect(1:length(tree.elements))
    tree_correct_exemplars_inds = _tree_elements_to_indicies(tree, elements)
    return (exemplars, tree_correct_exemplars_inds)
end

function exemplars_pam(
        encoded_states::Matrix{Float32},
        n_clusters::Int,
        distance_premetric::Distances.PreMetric
    ) :: Tuple{Vector{Int}, TreeNode}

    distances = Distances.pairwise(distance_premetric, encoded_states)
    exemplars = Clustering.kmedoids(distances, n_clusters).medoids
    distances_of_exemplars = Distances.pairwise(distance_premetric, encoded_states[:, exemplars])

    # create tree with normal hclust
    clustering = Clustering.hclust(distances_of_exemplars, linkage=:average)
    tree = _create_tree_hclust(clustering.merges)
    return (exemplars, tree)
end

function exemplars_kmedoids(
    encoded_states::Matrix{Float32},
    n_clusters::Int,
    distance_premetric::Distances.PreMetric
) :: Tuple{Vector{Int}, TreeNode}

    distances = Distances.pairwise(distance_premetric, encoded_states)
    exemplars = Clustering.kmedoids(distances, n_clusters).medoids
    distances_of_exemplars = Distances.pairwise(distance_premetric, encoded_states[:, exemplars])

    # create tree with normal hclust
    clustering = Clustering.hclust(distances_of_exemplars, linkage=:average)
    tree = _create_tree_hclust(clustering.merges)
    return (exemplars, tree)
end

# ------------------------------------------------------------------------------------------
# other functions

# used in genie exemplars
function _tree_elements_to_indicies(node::TreeNode, elements::Vector{Int}) :: TreeNode
    if is_leaf(node)
        new_node = TreeNode(nothing, nothing, elements)
        return new_node
    end
    elements_left = elements[1:length(node.left.elements)]
    elements_right = elements[(length(node.left.elements)+1):end]
    left = _tree_elements_to_indicies(node.left, elements_left)
    right = _tree_elements_to_indicies(node.right, elements_right)
    new_node = TreeNode(left, right, elements)

    return new_node
end

# functions exclusively for genie clust
function _create_exemplar_tree_number(children::Matrix{Int}, encoded_states::Matrix{Float32}, n_clusters, distance_premetric::Distances.PreMetric) :: TreeNode
    n_states = size(encoded_states, 2)
    clusters = Vector{TreeNode}(undef, n_states*2 - 1)
    for i in 1:n_states
        clusters[i] = TreeNode(nothing, nothing, [i])
    end
    last_index = n_states + 1

    index_first_real_cluster = n_states - n_clusters + 1

    for i in 1:(n_states - 1)
        left = clusters[children[i, 1]]
        right = clusters[children[i, 2]]

        if i < index_first_real_cluster
            clusters[last_index] = TreeNode(nothing, nothing, vcat(left.elements, right.elements))
        else
            if is_leaf(left)
                left = _create_node_exemplar(left, encoded_states, distance_premetric)
            end
            if is_leaf(right)
                right = _create_node_exemplar(right, encoded_states, distance_premetric)
            end
            clusters[last_index] = TreeNode(left, right, vcat(left.elements, right.elements))
        end
        last_index += 1
    end
    last_index -= 1

    @assert length(clusters[last_index].elements) == n_clusters

    return clusters[last_index]
end

function _create_node_exemplar(node::TreeNode, encoded_states::Matrix{Float32}, distance_premetric::Distances.PreMetric) :: TreeNode
    elements::Vector{Int} = node.elements
    elements_distances = Distances.pairwise(distance_premetric, encoded_states[:, elements])
    sum_of_distances = vec(sum(elements_distances, dims=1))
    exemplar = elements[argmin(sum_of_distances)]
    node_exemplar = TreeNode(nothing, nothing, [exemplar])
    return node_exemplar
end
