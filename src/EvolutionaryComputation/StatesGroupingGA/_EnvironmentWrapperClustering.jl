
# K-means of K-medoids based solution, now I experiment with genie clustering
# function _get_exemplars(encoded_states::Matrix{Float32}) :: Vector{Int}
#     # previous Affinity Propagation approach

#     # simmilarity_matrix = _cosine_simmilarity(encoded_states)

#     # # Affinity Propagation
#     # affinity_prop_result_median = AffinityPropagation.affinityprop(simmilarity_matrix; display=:iter)

#     # return affinity_prop_result_median.exemplars

#     # X-means approach

#     # xmeans = ClusteringHML.Xmeans()
#     # # it requires (samples_n, features_n) matrix, so I have to transpose it
#     # transposed_states = Matrix{Float64}(encoded_states')
#     # display(transposed_states)
#     # println("start Clustering")
#     # ClusteringHML.fit!(xmeans, transposed_states)
#     # centers = xmeans.centers
#     # scatter(centers[:, 1], centers[:, 2], legend=false, color=:green)  # Creates the scatter plot
#     # display(xmeans.centers)
#     # display(xmeans.labels)
#     # divide into groups based on labels
#     # Threads.@threads for i in 1:length(xmeans.centers)
#     # for i in 1:length(centers)
#     #     @inbounds begin
#     #         center = centers[i]
#     #         exemplars_ids[i] = argmin([sum((encoded_states[j] .- center).^2) for j in 1:length(labels) if labels[j] == i])
#     #     end
#     # end

#     # return exemplars_ids

#     scatter(encoded_states[1, :], encoded_states[2, :], legend=false)  # Creates the scatter plot

#     sqrt_n = sqrt(size(encoded_states, 2))
#     possible_k = sqrt_n .* [1.0]  # collect(0.25:0.25:2.0)  [0.25, 0.5, 1.0]
#     possible_k = [trunc(Int, k) for k in possible_k if k > 0]

#     state_wise_distances = _euclidean_distance(encoded_states, encoded_states)

#     # K-means approach
#     # k_results = Vector{Clustering.KmeansResult}(undef, length(possible_k))
#     # silhouettes = zeros(Float32, length(possible_k))
#     # Threads.@threads for i in 1:length(possible_k)
#     # # for i in 1:length(possible_k)
#     #     k = possible_k[i]
#     #     k_results[i] = Clustering.kmeans(encoded_states, k)
#     #     silhouettes[i] = Statistics.mean(Clustering.silhouettes(Clustering.assignments(k_results[i]), state_wise_distances))
#     # end
#     # labels = Clustering.assignments(k_result)
#     # exemplars_ids = zeros(Int, size(centers, 2))
#     # # divide into groups based on labels
#     # Threads.@threads for i in 1:size(centers, 2)
#     # # for i in 1:size(centers, 2)
#     #     @inbounds begin
#     #         center = centers[:, i]
#     #         exemplars_ids[i] = argmin([sum((encoded_states[:, j] .- center).^2) for j in 1:length(labels) if labels[j] == i])
#     #     end
#     # end
#     # k_result = k_results[argmax(silhouettes)]
#     # centers = k_result.centers

#     # scatter!(centers[1, :], centers[2, :], legend=false, color=:red)  # Adds the exemplars to the scatter plot


#     # K-medoids approach
#     k_results = Vector{Clustering.KmedoidsResult}(undef, length(possible_k))
#     silhouettes = zeros(Float32, length(possible_k))
#     Threads.@threads for i in 1:length(possible_k)
#     # for i in 1:length(possible_k)
#         k = possible_k[i]
#         k_results[i] = Clustering.kmedoids(state_wise_distances, k)
#         silhouettes[i] = Statistics.mean(Clustering.silhouettes(Clustering.assignments(k_results[i]), state_wise_distances))
#     end

#     k_result = k_results[argmax(silhouettes)]
#     exemplars_ids = k_result.medoids
#     scatter!(encoded_states[1, exemplars_ids], encoded_states[2, exemplars_ids], legend=false, color=:red)  # Adds the exemplars to the scatter plot

#     # Saving the plot to a file, e.g., PNG format
#     savefig("scatter_plot.png")
#     println("should be scattered now")

#     plot(possible_k, silhouettes, legend=false, xlabel="k", ylabel="Silhouette score", title="Silhouette score for different k values")
#     savefig("silhouette_score.png")

#     # display(exemplars_ids)

#     # _fill_diagonal!(state_wise_distances, 0.5)
#     # affinityprop = Clustering.affinityprop((-1) .* state_wise_distances; display=:iter)
#     # exemplars_ids = affinityprop.exemplars
#     # println("exemplars number $(length(exemplars_ids))")

#     return exemplars_ids
# end

function _get_exemplars(encoded_states::Matrix{Float32}) :: Tuple{Vector{Int}, TreeNode}
    # change encoded states to pyobject
    encoded_states_py = PyCall.PyObject(encoded_states')
    println(encoded_states_py.shape)
    display(encoded_states_py)
    # get genie clustering
    genie = genieclust.Genie(compute_full_tree=true, compute_all_cuts=true, verbose=true)
    @time genie.fit(encoded_states_py)

    children = collect(Array(genie.children_)')
    distances = Array(genie.distances_)
    
    sampled_indices = rand(1:size(encoded_states, 2), trunc(Int, sqrt(size(encoded_states, 2))))
    @time sampled_distance_matrix = _euclidean_distance(encoded_states[:, sampled_indices])
    @time median_distance = Statistics.median(sampled_distance_matrix)

    tree = _create_exemplar_tree(children, encoded_states, distances, size(encoded_states, 2), Float32(median_distance))
    return (tree.elements, tree)
end

function _create_exemplar_tree(children::Matrix{Int}, encoded_states::Matrix{Float32}, distances_clusters::Vector{Float32}, n_states::Int, distance_threshold::Float32) :: TreeNode
    clusters = Vector{TreeNode}(undef, n_states*2)
    for i in 1:n_states
        clusters[i] = TreeNode(nothing, nothing, 0.0, [i])
    end
    last_index = n_states

    for i in 1:size(children, 1)
        left = clusters[children[i, 1] + 1]
        right = clusters[children[i, 2] + 1]
        distance = distances_clusters[i]

        if distance < distance_threshold
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

    return clusters[last_index]
end

function _create_node_exemplar(node::TreeNode, encoded_states::Matrix{Float32}) :: Int
    elements::Vector{Int} = node.elements
    elements_distances = _euclidean_distance(encoded_states[:, elements])
    sum_of_distances = vec(sum(elements_distances, dims=1))
    exemplar = elements[argmin(sum_of_distances)]
    node_exemplar = TreeNode(nothing, nothing, node.distance, [exemplar])
    return node_exemplar
end

function _euclidean_distance(states::Matrix{Float32}) :: Matrix{Float32}
    return Distances.pairwise(Distances.Euclidean(), states)
end

function _euclidean_distance(states1::Matrix{Float32}, states2::Matrix{Float32}) :: Matrix{Float32}
    return Distances.pairwise(Distances.Euclidean(), states1, states2)
end

function _cosine_simmilarity(states::Matrix{Float32}) :: Matrix{Float32}
    # calculate cosine simmilarity
    simmilarity_matrix = states' * states
    sqrt_sum = sqrt.(sum(states .^ 2, dims=1))
    sqrt_matrix = sqrt_sum' * sqrt_sum
    simmilarity_matrix = simmilarity_matrix ./ sqrt_matrix

    return simmilarity_matrix
end

function _fill_diagonal!(matrix::Matrix, fill_quantile::Float64)
    values_not_diagonal = [matrix[i, j] for i in 1:size(matrix, 1), j in 1:size(matrix, 2) if i != j]
    diagonal_fill = Statistics.quantile(values_not_diagonal, fill_quantile)
    # along the diagonal fill median
    for i in 1:size(matrix, 1)
        matrix[i, i] = diagonal_fill
    end
end

function _cosine_simmilarity(states1::Matrix{Float32}, states2::Matrix{Float32}) :: Matrix{Float32}
    # calculate cosine simmilarity
    simmilarity_matrix = states1' * states2
    sqrt_sum1 = sqrt.(sum(states1 .^ 2, dims=1))
    sqrt_sum2 = sqrt.(sum(states2 .^ 2, dims=1))
    sqrt_matrix = sqrt_sum1' * sqrt_sum2
    simmilarity_matrix = simmilarity_matrix ./ sqrt_matrix

    return simmilarity_matrix
end