# ------------------------------------------------------------------------------------------
# time distance trees
# ------------------------------------------------------------------------------------------

function get_time_distance_tree(
    membership_states_trajectories::Vector{Matrix{Float32}};
    method::Symbol = :markov,  # :markov_fundamental, :mine
    hclust_time::Symbol = :average,  # :single, :complete, :average, :ward
) :: TreeNode
    if method == :markov
        tree = _create_time_distance_tree_markov_fundamental(membership_states_trajectories, hclust_time)
    elseif method == :mine
        tree = _create_time_distance_tree_mine(membership_states_trajectories, hclust_time)
    else
        throw(ArgumentError("Unknown method: $method"))
    end
    return tree
end

# ------------------------------------------------------------------------------------------
# Markov fundamental based time distance tree
# Read: https://lips.cs.princeton.edu/the-fundamental-matrix-of-a-finite-markov-chain/
function _create_time_distance_tree_markov_fundamental(
    membership_states_trajectories::Vector{Matrix{Float32}},
    hclust_time::Symbol = :average  # :single, :complete, :average, :ward
) :: TreeNode
    transition_matrix = _markov_transition_matrix(membership_states_trajectories)  # each row states for probability for going from row index state to column index state
    states_n = size(transition_matrix, 2)

    # w = P*w
    w_vec = LinearAlgebra.nullspace(transition_matrix - LinearAlgebra.I)
    w_vec = abs.(w_vec)
    w_vec ./= sum(w_vec)
    # w_vec ends up as a stationary distribution, it is one that is "stable", it is a distribution, hence abs and / sum
    W_matrix = repeat(w_vec, 1, states_n)  # all columns of W_matrix are the same and equal to w_vec
    N_fundamental_matrix = inv(LinearAlgebra.I - transition_matrix + W_matrix)  # N = (I - P + W)^-1

    # mean first passage time matrix - m_ij = expected time to get to state j from state i
    MFPT_matrix = Matrix{Float32}(undef, states_n, states_n)
    for j in 1:states_n, i in 1:states_n
        # Idk if I should go through columns or rows - what it actually represent?
        # either way at the end to calculate distance I average it along diagonal, so it doesn't matter here
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
    # println("\n\n\n transition_matrix: ")
    # display(transition_matrix)
    # println("\n\n\n w_vec: ")
    # display(w_vec)
    # println("\n\n\n W_matrix: ")
    # display(W_matrix)
    # println("\n\n\n mul1:")
    # display(W_matrix * transition_matrix)
    # println("\n\n\n mul2:")
    # display(transition_matrix * w_vec)
    # println("\n\n\n N_fundamental_matrix: ")
    # display(N_fundamental_matrix)
    # println("\n\n\n MFPT_matrix: ")
    # display(MFPT_matrix)
    # println("\n\n\n distances:")
    # display(distances)
    # throw("I guess I would like to end here")

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

    return transition_matrix
end



# ------------------------------------------------------------------------------------------
# my time distance tree

function _create_time_distance_tree_mine(
    membership_states_trajectories::Vector{Matrix{Float32}},
    hclust_time::Symbol = :average  # :single, :complete, :average, :ward
) :: TreeNode
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
