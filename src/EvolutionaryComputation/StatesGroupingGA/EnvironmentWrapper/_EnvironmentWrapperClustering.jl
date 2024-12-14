
function _get_exemplars(encoded_states::Matrix{Float32}, encoder::NeuralNetwork.AbstractNeuralNetwork, n_clusters::Int) :: Tuple{Vector{Int}, TreeNode}
    merge_clusters = GenieClust.genie_clust(encoded_states)
    tree = _create_exemplar_tree_number(merge_clusters, encoded_states, n_clusters)
    exemplars = tree.elements
    tree.elements = collect(1:length(tree.elements))
    _tree_elements_to_indicies!(tree)

    return (exemplars, tree)
end

function _tree_elements_to_indicies!(node::TreeNode)
    node.left.elements = node.elements[1:length(node.left.elements)]
    node.right.elements = node.elements[(length(node.left.elements)+1):end]
    
    if !is_leaf(node.left)
        _tree_elements_to_indicies!(node.left)
    end

    if !is_leaf(node.right)
        _tree_elements_to_indicies!(node.right)
    end
end

# function _get_distance_threshold(encoded_states::Matrix{Float32}) :: Float32
#     n_samples = 1000
#     # # sampled_indices = rand(1:size(encoded_states, 2), trunc(Int, sqrt(size(encoded_states, 2))))
#     sampled_indices = rand(1:size(encoded_states, 2), n_samples)
#     sampled_distance_matrix = _distance(encoded_states[:, sampled_indices])
    
    
#     median_distance = Statistics.median(sampled_distance_matrix)
#     println(median_distance)
#     # # median_distance = (distances_clusters[end] - distances_clusters[1]) / 2
#     # println("median distance: $median_distance")
#     # quantile = Statistics.quantile(vec(sampled_distance_matrix), 0.5)
#     # threshold = quantile
#     # pomysł - podwójnie użyć mediany, njpierw wybrać najbardziej środkowy punkt, potem medianę odleglości od niego do innych
#     sum_of_distances = [sum(column) for column in eachcol(sampled_distance_matrix)]
#     medoid = sampled_distance_matrix[:, argmin(sum_of_distances)]
#     medoid_median = Statistics.median(medoid)
#     println(medoid_median)
#     return medoid_median
# end

# function _create_exemplar_tree_distance(children::Matrix{Int}, encoded_states::Matrix{Float32}, n_states::Int, distance_threshold::Float32) :: TreeNode
#     clusters = Vector{TreeNode}(undef, n_states*2)
#     for i in 1:n_states
#         clusters[i] = TreeNode(nothing, nothing, 0.0, [i])
#     end
#     last_index = n_states + 1

#     for i in 1:size(children, 2)
#         left = clusters[children[1, i]]
#         right = clusters[children[2, i]]
#         distance = _median_distance(encoded_states[:, left.elements], encoded_states[:, right.elements])

#         if distance < distance_threshold && is_leaf(left) && is_leaf(right)
#             clusters[last_index] = TreeNode(nothing, nothing, distance, vcat(left.elements, right.elements))
#         else
#             if is_leaf(left)
#                 left = _create_node_exemplar(left, encoded_states)
#             end
#             if is_leaf(right)
#                 right = _create_node_exemplar(right, encoded_states)
#             end
#             clusters[last_index] = TreeNode(left, right, distance, vcat(left.elements, right.elements))
#         end
#         last_index += 1
#     end
#     last_index -= 1

#     return clusters[last_index]
# end

function _create_exemplar_tree_number(children::Matrix{Int}, encoded_states::Matrix{Float32}, n_clusters) :: TreeNode
    n_states = size(encoded_states, 2)
    clusters = Vector{TreeNode}(undef, n_states*2)
    for i in 1:n_states
        clusters[i] = TreeNode(nothing, nothing, 0.0, [i])
    end
    last_index = n_states + 1

    index_first_real_cluster = size(children, 2) - n_clusters + 2

    for i in 1:size(children, 2)
        left = clusters[children[1, i]]
        right = clusters[children[2, i]]
        distance = 0.0

        if i < index_first_real_cluster
            clusters[last_index] = TreeNode(nothing, nothing, distance, vcat(left.elements, right.elements))
        else
            if is_leaf(left)
                left = _create_node_exemplar(left, encoded_states)
            end
            if is_leaf(right)
                right = _create_node_exemplar(right, encoded_states)
            end
            clusters[last_index] = TreeNode(left, right, distance, vcat(left.elements, right.elements))
        end
        last_index += 1
    end
    last_index -= 1

    return clusters[last_index]
end

function _median_distance(states1::Matrix{Float32}, states2::Matrix{Float32}) :: Float64
    distances = _distance(states1, states2)
    mean = Statistics.median(distances)
    return mean
end

function _create_node_exemplar(node::TreeNode, encoded_states::Matrix{Float32}) :: TreeNode
    elements::Vector{Int} = node.elements
    elements_distances = _distance(encoded_states[:, elements])
    sum_of_distances = vec(sum(elements_distances, dims=1))
    exemplar = elements[argmin(sum_of_distances)]
    node_exemplar = TreeNode(nothing, nothing, node.distance, [exemplar])
    return node_exemplar
end

function _distance(states::Matrix{Float32}) :: Matrix{Float32}
    # return Distances.pairwise(Distances.Euclidean(), states)
    return Distances.pairwise(Distances.CosineDist(), states)
end

function _distance(states1::Matrix{Float32}, states2::Matrix{Float32}) :: Matrix{Float32}
    # return Distances.pairwise(Distances.Euclidean(), states1, states2)
    return Distances.pairwise(Distances.CosineDist(), states1, states2)
end
