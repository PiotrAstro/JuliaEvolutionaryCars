module StatesGrouping

import Distances
import Clustering
import Statistics
import LoopVectorization
import LinearAlgebra

import ..GenieClust
import ..PAM
import ..FCM

export get_exemplars, TreeNode, is_leaf, create_time_distance_tree_mine, create_time_distance_tree_markov_fundamental

# ------------------------------------------------------------------------------------------
# TreeNode for hierarchical clustering
# ------------------------------------------------------------------------------------------

struct TreeNode
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    elements::Vector{Int}
end


function is_leaf(tree::TreeNode) :: Bool
    return isnothing(tree.left) && isnothing(tree.right)
end


function _create_tree_hclust(merge_matrix::Matrix{Int}) :: TreeNode
    points_n = size(merge_matrix, 1) + 1
    leafs = Vector{TreeNode}(undef, points_n)
    clusters = Vector{TreeNode}(undef, points_n - 1)

    for i in 1:points_n
        leafs[i] = TreeNode(nothing, nothing, [i])
    end
    last_index = 1

    for i in 1:size(merge_matrix, 1)
        left_index = merge_matrix[i, 1]
        right_index = merge_matrix[i, 2]

        left = left_index >= 0 ? clusters[left_index] : leafs[-left_index]
        right = right_index >= 0 ? clusters[right_index] : leafs[-right_index]

        clusters[last_index] = TreeNode(left, right, vcat(left.elements, right.elements))
        last_index += 1
    end
    last_index -= 1

    return clusters[last_index]
end




# ------------------------------------------------------------------------------------------
# getting exemplars
# ------------------------------------------------------------------------------------------
import Plots
function get_exemplars(
        encoded_states::Matrix{Float32},
        n_clusters::Int;
        distance_metric::Symbol, # :cosine or :euclidean or cityblock
        exemplars_clustering::Symbol, # :genie or :pam or :kmedoids
        hclust_distance::Symbol, # :complete or :average or :single or :ward, ignored for genie
        mval::Int=2  # Fuzzy parameter, it is different that one in papers!

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
        return exemplars_pam(encoded_states, n_clusters, distance_premetric, hclust_distance)
    elseif exemplars_clustering == :kmedoids
        return exemplars_kmedoids(encoded_states, n_clusters, distance_premetric, hclust_distance)
    elseif exemplars_clustering == :fcm_rand
        return exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :rand)
    elseif exemplars_clustering == :fcm_best
        return exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :best)
    elseif exemplars_clustering == :fcm_best_rand
        return exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :best_rand)
    else
        throw(ArgumentError("Unknown exemplars_clustering: $exemplars_clustering"))
    end


    # kmedoids = PyCall.pyimport("kmedoids")

    # @time begin
    #     distances = PyCall.PyObject(Distances.pairwise(distance_premetric, encoded_states)')
    #     exp_python = Vector{Int}(kmedoids.fasterpam(distances, n_clusters, max_iter=100, init="random", n_cpu=1).medoids)
    # end
    # println("\n\nGenie:")
    # @time exp_genie, _ = exemplars_genie(encoded_states, n_clusters, distance_premetric)
    # println("\n\nPAM:")
    # @time exp_pam, _ = exemplars_pam(encoded_states, n_clusters, distance_premetric, hclust_distance)
    # println("\n\nKmedoids:")
    # @time exp_kmedoids, _ = exemplars_kmedoids(encoded_states, n_clusters, distance_premetric, hclust_distance)
    # println("\n\nFCM rand:")
    # @time exp_fcm_rand, _ = exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :rand)
    # println("\n\nFCM best:")
    # @time exp_fcm_best, _ = exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :best)
    # println("\n\nFCM best rand:")
    # @time exp_fcm_best_rand, _ = exemplars_fcm(encoded_states, n_clusters, distance_premetric, hclust_distance, mval, :best_rand)

    # println("n_clusters: $(n_clusters)\nn_states: $(size(encoded_states, 2))")
    # Plots.scatter(encoded_states[1, :], encoded_states[2, :], label="States", size=(1500, 1500), markerstrokewidth = 0.1, color=:blue)
    # Plots.scatter!(encoded_states[1, exp_genie], encoded_states[2, exp_genie], label="Exemplars_genie", markerstrokewidth = 0.1; color=:red)
    # Plots.scatter!(encoded_states[1, exp_pam], encoded_states[2, exp_pam], label="Exemplars_pam", markerstrokewidth = 0.1, color=:pink)
    # Plots.scatter!(encoded_states[1, exp_kmedoids], encoded_states[2, exp_kmedoids], label="Exemplars_kmedoids", markerstrokewidth = 0.1, color=:purple)
    # # Plots.scatter!(encoded_states[1, exp_python], encoded_states[2, exp_python], label="Exemplars_python", markerstrokewidth = 0.1, color=:orange)
    # Plots.scatter!(encoded_states[1, exp_fcm_rand], encoded_states[2, exp_fcm_rand], label="Exemplars_fcm_rand", markerstrokewidth = 0.1, color=:green)
    # Plots.scatter!(encoded_states[1, exp_fcm_best], encoded_states[2, exp_fcm_best], label="Exemplars_fcm_best", markerstrokewidth = 0.1, color=:yellow)
    # Plots.scatter!(encoded_states[1, exp_fcm_best_rand], encoded_states[2, exp_fcm_best_rand], label="Exemplars_fcm_best_rand", markerstrokewidth = 0.1, color=:brown)
    # Plots.savefig("log/encoded_states_pam_genie_kmedoids_fcm.png")
    # throw("I guess I would like to end here")
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
    return exemplars, tree_correct_exemplars_inds
end

function exemplars_pam(
        encoded_states::Matrix{Float32},
        n_clusters::Int,
        distance_premetric::Distances.PreMetric,
        hclust_distance::Symbol
    ) :: Tuple{Vector{Int}, TreeNode}

    distances = Distances.pairwise(distance_premetric, encoded_states)
    exemplars = PAM.faster_pam(distances, n_clusters)
    distances_of_exemplars = Distances.pairwise(distance_premetric, encoded_states[:, exemplars])

    # create tree with normal hclust
    clustering = Clustering.hclust(distances_of_exemplars, linkage=hclust_distance)
    tree = _create_tree_hclust(clustering.merges)
    return exemplars, tree
end

function exemplars_kmedoids(
        encoded_states::Matrix{Float32},
        n_clusters::Int,
        distance_premetric::Distances.PreMetric,
        hclust_distance::Symbol
    ) :: Tuple{Vector{Int}, TreeNode}

    distances = Distances.pairwise(distance_premetric, encoded_states)
    exemplars = Clustering.kmedoids(distances, n_clusters).medoids
    distances_of_exemplars = Distances.pairwise(distance_premetric, encoded_states[:, exemplars])

    # create tree with normal hclust
    clustering = Clustering.hclust(distances_of_exemplars, linkage=hclust_distance)
    tree = _create_tree_hclust(clustering.merges)
    return exemplars, tree
end

function exemplars_fcm(
        encoded_states::Matrix{Float32},
        n_clusters::Int,
        distance_premetric::Distances.PreMetric,
        hclust_distance::Symbol,
        mval :: Int,
        initialization_mode::Symbol
    ) :: Tuple{Vector{Int}, TreeNode}

    distances = Distances.pairwise(distance_premetric, encoded_states)
    exemplars = FCM.fuzzy_kmedoids(distances, n_clusters, mval; initialization=initialization_mode)
    distances_of_exemplars = Distances.pairwise(distance_premetric, encoded_states[:, exemplars])

    # create tree with normal hclust
    clustering = Clustering.hclust(distances_of_exemplars, linkage=hclust_distance)
    tree = _create_tree_hclust(clustering.merges)
    return exemplars, tree
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














# ------------------------------------------------------------------------------------------
# time distance trees
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
# Markov fundamental based time distance tree
# Read: https://lips.cs.princeton.edu/the-fundamental-matrix-of-a-finite-markov-chain/
function create_time_distance_tree_markov_fundamental(
        membership_states_trajectories::Vector{Matrix{Float32}},
        hclust_time::Symbol = :average  # :single, :complete, :average, :ward
    )
    transition_matrix = _markov_transition_matrix(membership_states_trajectories)  # each row states for probability for going from row index state to column index state
    states_n = size(transition_matrix, 2)

    # w = P*w
    w_vec = LinearAlgebra.nullspace(transition_matrix' - LinearAlgebra.I)  # we transpose matrix, because julia works this way here
    W_matrix = repeat(w_vec', states_n, 1)  # all rows of W_matrix are the same and equal to w_vec
    N_fundamental_matrix = inv(LinearAlgebra.I - transition_matrix + W_matrix)  # N = (I - P + W)^-1

    # mean first passage time matrix - m_ij = expected time to get to state j from state i
    MFPT_matrix = Matrix{Float32}(undef, states_n, states_n)
    for j in 1:states_n, i in 1:states_n
        MFPT_matrix[i, j] = (N_fundamental_matrix[j, j] - N_fundamental_matrix[i, j]) / w_vec[j]
    end

    # calculating distance matrix for hclust - MFPT_matrix is not symmetric, we have to transform it to such
    distances = Matrix{Float32}(undef, states_n, states_n)
    for i in 1:states_n
        for j in (i + 1):states_n
            value = (MFPT_matrix[i, j] + MFPT_matrix[j, i]) / 2
            distances[i, j] = value
            distances[j, i] = value
        end
        distances[i, i] = 0.0f0
    end

    # now we will just cluster it
    clustering = Clustering.hclust(distances, linkage=hclust_time)
    tree = _create_tree_hclust(clustering.merges)
    return tree
end

function _markov_transition_matrix(membership_states_trajectories::Vector{Matrix{Float32}})::Matrix{Float32}
    states_n = size(membership_states_trajectories[1], 1)
    transition_matrix = zeros(Float32, states_n, states_n)

    for memberships in membership_states_trajectories
        LoopVectorization.@turbo for step in 1:(size(memberships, 2) - 1)
            for current_state in 1:states_n
                for next_state in 1:states_n
                    # here we use transposed matrix, to make it more cache friendly for Julia
                    transition_matrix[next_state, current_state] += memberships[current_state, step] * memberships[next_state, step + 1]
                end
            end
        end
    end

    for col in eachcol(transition_matrix)
        col ./= sum(col)
    end

    transposed_matrix = collect(transition_matrix')
    return transposed_matrix
end









# ------------------------------------------------------------------------------------------
# my time distance tree

function create_time_distance_tree_mine(
        membership_states_trajectories::Vector{Matrix{Float32}},
        hclust_time::Symbol = :average  # :single, :complete, :average, :ward
    )
    exemplars_n = size(membership_states_trajectories[1], 1)

    trajectories_time_distances_by_exemplar = [Vector{Matrix{Int}}() for _ in 1:length(membership_states_trajectories)]
    Threads.@threads for i in 1:length(membership_states_trajectories)
    # for i in 1:length(encoded_states_trajectories)
        trajectories_time_distances_by_exemplar[i] = _create_time_distance_matrix(membership_states_trajectories[i])
    end

    # vector of matrices
    time_distances_by_exemplar = [
        (reduce(hcat, [trajectories_time_distances_by_exemplar[trajectory][exemplar_id] for trajectory in 1:length(membership_states_trajectories)]))
        for exemplar_id in 1:exemplars_n
    ]

    # distance for clustering
    distances = zeros(Float32, exemplars_n, exemplars_n)
    for i in 1:(exemplars_n - 1)
        for j in (i + 1):exemplars_n
            combined_distances = vcat(time_distances_by_exemplar[i][j, :], time_distances_by_exemplar[j][i, :])
            distance_value = Statistics.median(combined_distances)
            distances[i, j] = distance_value
            distances[j, i] = distance_value
        end
    end

    # display(distances)

    # used to be :single
    clustering = Clustering.hclust(distances, linkage=hclust_time)
    # Plots.plot(clustering, 
    #     title="Hierarchical Clustering Dendrogram",
    #     xlabel="Sample Index", 
    #     ylabel="Distance",
    #     linewidth=2,
    #     color=:blues,
    #     legend=false
    # )
    # Plots.savefig("log/dendrogram.png")
    tree = _create_tree_hclust(clustering.merges)
    return tree
end

function _create_time_distance_matrix(memberships::Matrix{Float32})::Vector{Matrix{Int}}
    exemplars_n = size(memberships, 1)
    trajectory_length = size(memberships, 2)

    # Get indices of closest exemplars for each state in the trajectory
    exemplars_indices_trajectory = _closest_exemplars_indices(memberships)

    # Build a list of occurrence times for each exemplar
    all_indices = [Int[] for _ in 1:exemplars_n]
    for (i, exemplar_id) in enumerate(exemplars_indices_trajectory)
        push!(all_indices[exemplar_id], i)
    end

    # Initialize the time distances by exemplar
    time_distances_by_exemplar = fill(trajectory_length, trajectory_length, exemplars_n)

    for exemplar_index in 1:exemplars_n
        occurrence_times = all_indices[exemplar_index]

        if !isempty(occurrence_times)
            # Initialize distances with maximum possible value
            distances = @view time_distances_by_exemplar[:, exemplar_index]

            # Set distances at occurrence times to zero
            distances[occurrence_times] .= 0

            # Forward pass to compute minimum distances
            previous_value = distances[1]
            for t in 2:trajectory_length
                previous_value = distances[t] = min(distances[t], previous_value + 1)
            end

            # Backward pass to refine minimum distances
            previous_value = distances[trajectory_length]
            for t in (trajectory_length - 1):-1:1
                previous_value = distances[t] = min(distances[t], previous_value + 1)
            end
        end
    end

    time_distances_separated_for_exemplars = Vector{Matrix{Int}}(undef, exemplars_n)
    transposed_time_distances_by_exemplar = time_distances_by_exemplar'
    for i in 1:exemplars_n
        if !isempty(all_indices[i])
            time_distances_separated_for_exemplars[i] = transposed_time_distances_by_exemplar[:, all_indices[i]]
        else
            time_distances_separated_for_exemplars[i] = fill(trajectory_length, exemplars_n, 1)
        end
    end

    return time_distances_separated_for_exemplars
end

function _closest_exemplars_indices(memberships::Matrix{Float32}) :: Vector{Int}
    closest_exemplars = [argmax(one_col) for one_col in eachcol(memberships)]
    return closest_exemplars
end



end