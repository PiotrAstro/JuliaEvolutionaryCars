
# ------------------------------------------------------------------------------------------
# getting exemplars
# ------------------------------------------------------------------------------------------
# import Plots
import InteractiveUtils
import BenchmarkTools
import Profile
import PProf
function get_exemplars(
    encoded_states::Matrix{Float32},
    n_clusters::Int;
    distance_metric::Symbol, # :cosine or :euclidean or cityblock
    exemplars_clustering::Symbol, # :genie or :pam or :kmedoids
)::Vector{Int}
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

    if exemplars_clustering == :genie
        return exemplars_genie(encoded_states, n_clusters, distance_premetric)
    elseif exemplars_clustering == :pam
        return exemplars_pam(encoded_states, n_clusters, distance_premetric)
    elseif exemplars_clustering == :my_pam_random
        return exemplars_my_pam(encoded_states, n_clusters, distance_premetric, :random)
    elseif exemplars_clustering == :my_pam_random_increasing
        return exemplars_my_pam(encoded_states, n_clusters, distance_premetric, :random_increasing)
    elseif exemplars_clustering == :my_pam_weighted_random
        return exemplars_my_pam(encoded_states, n_clusters, distance_premetric, :weighted_random)
    elseif exemplars_clustering == :kmedoids
        return exemplars_kmedoids(encoded_states, n_clusters, distance_premetric)
    elseif exemplars_clustering == :k_means_medoids
        return exemplars_k_means_medoids(encoded_states, n_clusters, distance_premetric)
    # elseif exemplars_clustering == :fcm_rand
    #     return exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :rand)
    # elseif exemplars_clustering == :fcm_best
    #     return exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :best)
    # elseif exemplars_clustering == :fcm_best_rand
    #     return exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :best_rand)
    else
        throw(ArgumentError("Unknown exemplars_clustering: $exemplars_clustering"))
    end


    # kmedoids = PyCall.pyimport("kmedoids")

    # @time begin
    #     distances = PyCall.PyObject(Distances.pairwise(distance_premetric, encoded_states)')
    #     exp_python = Vector{Int}(kmedoids.fasterpam(distances, n_clusters, max_iter=100, init="random", n_cpu=1).medoids)
    # end

    # smaller_encoded_states_for_compilation = encoded_states[:, 1:(5*n_clusters)]
    # exemplars_genie(smaller_encoded_states_for_compilation, n_clusters, distance_premetric)
    # exemplars_pam(smaller_encoded_states_for_compilation, n_clusters, distance_premetric)
    # exemplars_kmedoids(smaller_encoded_states_for_compilation, n_clusters, distance_premetric)
    # exemplars_k_means_medoids(smaller_encoded_states_for_compilation, n_clusters, distance_premetric)
    # exemplars_my_pam(smaller_encoded_states_for_compilation, n_clusters, distance_premetric, :random)
    # exemplars_my_pam(smaller_encoded_states_for_compilation, n_clusters, distance_premetric, :random_increasing)
    # exemplars_my_pam(smaller_encoded_states_for_compilation, n_clusters, distance_premetric, :weighted_random)


    # exemplars_fcm(smaller_encoded_states_for_compilation, n_clusters, distance_premetric, hclust_distance, mval, :rand)
    # exemplars_fcm(smaller_encoded_states_for_compilation, n_clusters, distance_premetric, hclust_distance, mval, :best)
    # exemplars_fcm(smaller_encoded_states_for_compilation, n_clusters, distance_premetric, hclust_distance, mval, :best_rand)


    # println("\n\nGenie:")
    # @time exp_genie = exemplars_genie(encoded_states, n_clusters, distance_premetric)
    # # display(BenchmarkTools.@benchmark exemplars_genie($encoded_states, $n_clusters, $distance_premetric))
    # println("\n\nLoss:", _loss(encoded_states, exp_genie, distance_premetric))

    # println("\n\nPAM:")
    # @time exp_pam = exemplars_pam(encoded_states, n_clusters, distance_premetric)
    # # display(BenchmarkTools.@benchmark exemplars_pam($encoded_states, $n_clusters, $distance_premetric))
    # println("\n\nLoss:", _loss(encoded_states, exp_pam, distance_premetric))

    # println("\n\nKmedoids:")
    # @time exp_kmedoids = exemplars_kmedoids(encoded_states, n_clusters, distance_premetric)
    # # display(BenchmarkTools.@benchmark exemplars_kmedoids($encoded_states, $n_clusters, $distance_premetric))
    # println("\n\nLoss:", _loss(encoded_states, exp_kmedoids, distance_premetric))

    # println("\n\nK_means_medoids:")
    # @time exp_k_means_medoids = exemplars_k_means_medoids(encoded_states, n_clusters, distance_premetric)
    # # display(BenchmarkTools.@benchmark exemplars_k_means_medoids($encoded_states, $n_clusters, $distance_premetric))
    # println("\n\nLoss:", _loss(encoded_states, exp_k_means_medoids, distance_premetric))

    # println("\n\nMy PAM random:")
    # @time exp_my_pam_ran = exemplars_my_pam(encoded_states, n_clusters, distance_premetric, :random)
    # # display(BenchmarkTools.@benchmark exemplars_my_pam($encoded_states, $n_clusters, $distance_premetric, :random))
    # println("\n\nLoss:", _loss(encoded_states, exp_my_pam_ran, distance_premetric))

    # println("\n\nMy PAM random increasing:")
    # @time exp_my_pam_ran_inc = exemplars_my_pam(encoded_states, n_clusters, distance_premetric, :random_increasing)
    # # display(BenchmarkTools.@benchmark exemplars_my_pam($encoded_states, $n_clusters, $distance_premetric, :random_increasing))
    # println("\n\nLoss:", _loss(encoded_states, exp_my_pam_ran_inc, distance_premetric))

    # println("\n\nMy PAM weighted random:")
    # @time exp_my_pam_weighted_random = exemplars_my_pam(encoded_states, n_clusters, distance_premetric, :weighted_random)
    # # display(BenchmarkTools.@benchmark exemplars_my_pam($encoded_states, $n_clusters, $distance_premetric, :weighted_random))
    # println("\n\nLoss:", _loss(encoded_states, exp_my_pam_weighted_random, distance_premetric))

    # Profile.clear()
    # Profile.@profile for _ in 1:10
    #     exp_my_pam = exemplars_my_pam(encoded_states, n_clusters, distance_premetric, :random_increasing)
    # end
    # PProf.pprof(;webport=2137)

    # Profile.clear()
    # Profile.@profile for _ in 1:10
    #     exp_my_pam = exemplars_my_pam(encoded_states, n_clusters, distance_premetric, :best)
    # end
    # PProf.pprof(;webport=1111)

    # throw("I guess I would like to end here")


    # println("\n\nFCM rand:")
    # @time exp_fcm_rand, _ = exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :rand)
    # println("\n\nFCM best:")
    # @time exp_fcm_best, _ = exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :best)
    # println("\n\nFCM best rand:")
    # @time exp_fcm_best_rand, _ = exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :best_rand)

    # println("n_clusters: $(n_clusters)\nn_states: $(size(encoded_states, 2))")
    # Plots.scatter(encoded_states[1, :], encoded_states[2, :], label="States", size=(1500, 1500), markerstrokewidth = 0.1, color=:aqua)
    # Plots.scatter!(encoded_states[1, exp_genie], encoded_states[2, exp_genie], label="Exemplars_genie", markerstrokewidth = 0.1; color=:red)
    # Plots.scatter!(encoded_states[1, exp_pam], encoded_states[2, exp_pam], label="Exemplars_pam", markerstrokewidth = 0.1, color=:pink)
    # Plots.scatter!(encoded_states[1, exp_kmedoids], encoded_states[2, exp_kmedoids], label="Exemplars_kmedoids", markerstrokewidth = 0.1, color=:purple)
    # Plots.scatter!(encoded_states[1, exp_k_means_medoids], encoded_states[2, exp_k_means_medoids], label="Exemplars_k_means_medoids", markerstrokewidth = 0.1, color=:blue)
    # Plots.scatter!(encoded_states[1, exp_my_pam], encoded_states[2, exp_my_pam], label="Exemplars_my_pam", markerstrokewidth = 0.1, color=:green)

    # # Plots.scatter!(encoded_states[1, exp_python], encoded_states[2, exp_python], label="Exemplars_python", markerstrokewidth = 0.1, color=:orange)
    # Plots.scatter!(encoded_states[1, exp_fcm_rand], encoded_states[2, exp_fcm_rand], label="Exemplars_fcm_rand", markerstrokewidth = 0.1, color=:green)
    # Plots.scatter!(encoded_states[1, exp_fcm_best], encoded_states[2, exp_fcm_best], label="Exemplars_fcm_best", markerstrokewidth = 0.1, color=:yellow)
    # Plots.scatter!(encoded_states[1, exp_fcm_best_rand], encoded_states[2, exp_fcm_best_rand], label="Exemplars_fcm_best_rand", markerstrokewidth = 0.1, color=:brown)
    
    # Plots.savefig("log/encoded_states_pam_genie_kmedoids_my_pam.png")
    # throw("I guess I would like to end here")
end



# ------------------------------------------------------------------------------------------
# exemplars functions

function exemplars_genie(
    encoded_states::Matrix{Float32},
    n_clusters::Int,
    distance_premetric::Distances.SemiMetric
)::Vector{Int}
    merge_clusters = GenieClust.genie_clust(encoded_states; distance_callable=distance_premetric)
    tree = _create_exemplar_tree_number(merge_clusters, encoded_states, n_clusters, distance_premetric)
    exemplars = tree.elements
    return exemplars
end

function exemplars_pam(
    encoded_states::Matrix{Float32},
    n_clusters::Int,
    distance_premetric::Distances.SemiMetric
)::Vector{Int}

    distances = Distances.pairwise(distance_premetric, encoded_states)
    exemplars = PAM.faster_pam(distances, n_clusters)
    return exemplars
end

# function exemplars_bandit_pam(
#     encoded_states::Matrix{Float32},
#     n_clusters::Int,
#     distance_premetric::Distances.SemiMetric
# )::Vector{Int}
#     exemplars = BanditPAM.bandit_pam_pp(encoded_states, distance_premetric, n_clusters)
#     return exemplars
# end

function exemplars_my_pam(
    encoded_states::Matrix{Float32},
    n_clusters::Int,
    distance_premetric::Distances.SemiMetric,
    candidates_method=:random
)::Vector{Int}
    exemplars = MyPAM.my_pam(encoded_states, distance_premetric, n_clusters, candidates_method=candidates_method)
    return exemplars
end

function exemplars_kmedoids(
    encoded_states::Matrix{Float32},
    n_clusters::Int,
    distance_premetric::Distances.SemiMetric
)::Vector{Int}

    distances = Distances.pairwise(distance_premetric, encoded_states)
    exemplars = Clustering.kmedoids(distances, n_clusters).medoids
    return exemplars
end

function exemplars_k_means_medoids(
    encoded_states::Matrix{Float32},
    n_clusters::Int,
    distance_premetric::Distances.SemiMetric
)::Vector{Int}
    # calculate k-means
    k_means_assignments = Logging.with_logger(Logging.NullLogger()) do
        Clustering.kmeans(encoded_states, n_clusters; distance=distance_premetric).assignments
    end

    # get best exemplars within clusters
    exemplars = Vector{Int}(undef, n_clusters)
    for cluster_id in 1:n_clusters
        points_in_cluster_tmp = encoded_states[:, k_means_assignments.==cluster_id]
        distances_tmp = Distances.pairwise(distance_premetric, points_in_cluster_tmp)
        exemplars[cluster_id] = argmin(sum(dist_col) for dist_col in eachcol(distances_tmp))
    end
    return exemplars
end

# It was pretty bad, so I do not use it here
# function exemplars_fcm(
#         encoded_states::Matrix{Float32},
#         n_clusters::Int,
#         distance_premetric::Distances.SemiMetric,
#         hclust_distance::Symbol,
#         mval :: Int,
#         initialization_mode::Symbol
#     ) :: Tuple{Vector{Int}, TreeNode}

#     distances = Distances.pairwise(distance_premetric, encoded_states)
#     exemplars = FCM.fuzzy_kmedoids(distances, n_clusters, mval; initialization=initialization_mode)
#     distances_of_exemplars = Distances.pairwise(distance_premetric, encoded_states[:, exemplars])

#     # create tree with normal hclust
#     clustering = Clustering.hclust(distances_of_exemplars, linkage=hclust_distance)
#     tree = _create_tree_hclust(clustering.merges)
#     return exemplars, tree
# end

# ------------------------------------------------------------------------------------------
# other functions

function _loss(encoded_states::Matrix{Float32}, exemplars::Vector{Int}, distance_premetric::Distances.SemiMetric)::Float32
    distances = Distances.pairwise(distance_premetric, encoded_states[:, exemplars], encoded_states)
    loss = sum(minimum(col) for col in eachcol(distances))
    return loss
end

# used in genie exemplars
function _tree_elements_to_indicies(node::TreeNode, elements::Vector{Int})::TreeNode
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
function _create_exemplar_tree_number(children::Matrix{Int}, encoded_states::Matrix{Float32}, n_clusters, distance_premetric::Distances.PreMetric)::TreeNode
    n_states = size(encoded_states, 2)
    clusters = Vector{TreeNode}(undef, n_states * 2 - 1)
    for i in 1:n_states
        clusters[i] = TreeNode(nothing, nothing, [i])
    end
    last_index = n_states + 1

    index_first_real_cluster = n_states - n_clusters + 1

    for i in 1:(n_states-1)
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

function _create_node_exemplar(node::TreeNode, encoded_states::Matrix{Float32}, distance_premetric::Distances.PreMetric)::TreeNode
    elements::Vector{Int} = node.elements
    elements_distances = Distances.pairwise(distance_premetric, encoded_states[:, elements])
    sum_of_distances = vec(sum(elements_distances, dims=1))
    exemplar = elements[argmin(sum_of_distances)]
    node_exemplar = TreeNode(nothing, nothing, [exemplar])
    return node_exemplar
end
