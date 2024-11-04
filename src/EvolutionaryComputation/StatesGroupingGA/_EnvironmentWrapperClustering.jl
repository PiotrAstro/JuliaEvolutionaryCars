
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

#     state_wise_distances = _distance(encoded_states, encoded_states)

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

function _get_exemplars(encoded_states::Matrix{Float32}, encoder::NeuralNetwork.AbstractNeuralNetwork) :: Tuple{Vector{Int}, TreeNode}
    # change encoded states to pyobject
    encoded_states_py = PyCall.PyObject(encoded_states')
    # get genie clustering
    n_clusters = 200  # 200 - works well, can get optimal solution  # size(encoded_states, 2)  # 100
    genie = genieclust.Genie(n_clusters=n_clusters, compute_full_tree=true, verbose=true, gini_threshold=0.05, affinity="cosine")
    @time genie.fit(encoded_states_py)

    children = collect(Array(genie.children_)')
    distances_clusters = Array(genie.distances_)

    # threshold = _get_distance_threshold(encoded_states)

    tree = _create_exemplar_tree_number(children, encoded_states, distances_clusters, size(encoded_states, 2), n_clusters)
    # tree = _create_exemplar_tree_distance(children, encoded_states, distances_clusters, size(encoded_states, 2), Float32(threshold))
    exemplars = tree.elements
    tree.elements = collect(1:length(tree.elements))
    _tree_elements_to_indicies!(tree)
    
    # n_clusters = trunc(Int, sqrt(size(encoded_states, 2)))
    # n_clusters = 30
    # tree = _create_exemplar_tree_number(children, encoded_states, distances_clusters, size(encoded_states, 2), n_clusters)

    # Plots.scatter(encoded_states[1, :], encoded_states[2, :], legend=false, marker=:x, size=(3000, 3000))  # Creates the scatter plot
    Plots.scatter(encoded_states[1, 1:2], encoded_states[2, 1:2], legend=false, marker=:x, size=(3000, 3000))
    # Plots.scatter!(encoded_states[1, exemplars], encoded_states[2, exemplars], legend=false, color=:red)  # Adds the exemplars to the scatter plot
    # load arrays of states from log/states_(number).jld and encode them and plot, form states 1 to 9
    for i in 1:9
        states = JLD.load("log/states_$i.jld")["states"]
        encoded_states_loaded = NeuralNetwork.predict(encoder, states)
        Plots.scatter!(encoded_states_loaded[1, :], encoded_states_loaded[2, :], legend=false, markerstrokewidth=0)
    end
    
    timestamp_string = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    Plots.savefig("log/scatter_plot_$timestamp_string.png")

    labels = Array(genie.labels_)
    labels_ids = [Vector{Int}() for _ in 1:n_clusters]
    for i in 1:length(labels)
       push!(labels_ids[labels[i] + 1], i) 
    end

    # display(labels_ids)
    # display(labels_ids[1])

    for cluster in labels_ids
        Plots.scatter!(encoded_states[1, cluster], encoded_states[2, cluster], legend=false, marker=:utriangle, markerstrokewidth=0, alpha=0.6)
    end
    Plots.savefig("log/scatter_plot_clusters_$timestamp_string.png")

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

function _get_distance_threshold(encoded_states::Matrix{Float32}) :: Float32
    n_samples = 1000
    # # sampled_indices = rand(1:size(encoded_states, 2), trunc(Int, sqrt(size(encoded_states, 2))))
    sampled_indices = rand(1:size(encoded_states, 2), n_samples)
    sampled_distance_matrix = _distance(encoded_states[:, sampled_indices])
    
    
    median_distance = Statistics.median(sampled_distance_matrix)
    println(median_distance)
    # # median_distance = (distances_clusters[end] - distances_clusters[1]) / 2
    # println("median distance: $median_distance")
    # quantile = Statistics.quantile(vec(sampled_distance_matrix), 0.5)
    # threshold = quantile
    # pomysł - podwójnie użyć mediany, njpierw wybrać najbardziej środkowy punkt, potem medianę odleglości od niego do innych
    sum_of_distances = [sum(column) for column in eachcol(sampled_distance_matrix)]
    medoid = sampled_distance_matrix[:, argmin(sum_of_distances)]
    medoid_median = Statistics.median(medoid)
    println(medoid_median)
    return medoid_median
end

function _create_exemplar_tree_distance(children::Matrix{Int}, encoded_states::Matrix{Float32}, distances_clusters::Vector{Float32}, n_states::Int, distance_threshold::Float32) :: TreeNode
    clusters = Vector{TreeNode}(undef, n_states*2)
    for i in 1:n_states
        clusters[i] = TreeNode(nothing, nothing, 0.0, [i])
    end
    last_index = n_states + 1

    for i in 1:size(children, 2)
        left = clusters[children[1, i] + 1]
        right = clusters[children[2, i] + 1]
        # distance = distances_clusters[i]
        distance = _median_distance(encoded_states[:, left.elements], encoded_states[:, right.elements])
        # println(distance)

        if distance < distance_threshold && is_leaf(left) && is_leaf(right)
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

function _create_exemplar_tree_number(children::Matrix{Int}, encoded_states::Matrix{Float32}, distances_clusters::Vector{Float32}, n_states::Int, n_clusters) :: TreeNode
    clusters = Vector{TreeNode}(undef, n_states*2)
    for i in 1:n_states
        clusters[i] = TreeNode(nothing, nothing, 0.0, [i])
    end
    last_index = n_states + 1

    index_first_real_cluster = size(children, 2) - n_clusters + 2

    for i in 1:size(children, 2)
        left = clusters[children[1, i] + 1]
        right = clusters[children[2, i] + 1]
        distance = distances_clusters[i]

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