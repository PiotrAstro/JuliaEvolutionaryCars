function _create_time_distance_tree(encoded_states_trajectories::Vector{Matrix{Float32}}, encoded_exemplars)
    exemplars_n = size(encoded_exemplars, 2)

    trajectories_time_distances_by_exemplar = [Vector{Matrix{Int}}() for _ in 1:length(encoded_states_trajectories)]
    # Threads.@threads for i in 1:length(encoded_states_trajectories)
    for i in 1:length(encoded_states_trajectories)
        trajectories_time_distances_by_exemplar[i] = _create_time_distance_matrix(encoded_states_trajectories[i], encoded_exemplars)
    end

    # vector of matrices
    time_distances_by_exemplar = [
        (reduce(hcat, [trajectories_time_distances_by_exemplar[trajectory][exemplar_id] for trajectory in 1:length(encoded_states_trajectories)]))
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
    clustering = Clustering.hclust(distances, linkage=:average)
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

function _create_time_distance_matrix(encoded_states_trajectory::Matrix{Float32}, encoded_exemplars::Matrix{Float32})::Vector{Matrix{Int}}
    exemplars_n = size(encoded_exemplars, 2)
    trajectory_length = size(encoded_states_trajectory, 2)

    # Get indices of closest exemplars for each state in the trajectory
    exemplars_indices_trajectory = _closest_exemplars_indices(encoded_states_trajectory, encoded_exemplars)

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

function _closest_exemplars_indices(encoded_states::Matrix{Float32}, exemplars::Matrix{Float32}) :: Vector{Int}
    distances_matrix = Distances.pairwise(Distances.CosineDist(), exemplars, encoded_states)
    closest_exemplars = [argmin(one_col) for one_col in eachcol(distances_matrix)]
    return closest_exemplars
end