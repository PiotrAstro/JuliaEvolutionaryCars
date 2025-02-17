module FCM
# Based on paper: Low-Complexity Fuzzy Relational Clustering Algorithms for Web Mining
# DOI: 10.1109/91.940971

import LoopVectorization
import Random
import LinearAlgebra

export fuzzy_kmedoids

function fuzzy_kmedoids(distances::Matrix{F}, k::Int, mval::Int; max_iter::Int=200, initialization::Symbol=:rand) :: Vector{Int} where {F<:AbstractFloat}  # Indicies
    # Initialization
    points_n = size(distances, 1)
    if k > points_n
        error("Number of clusters is greater than number of points")
    end

    if initialization == :rand
        indicies = _random_initialization(distances, k)
    elseif initialization == :best
        indicies = _best_initialization(distances, k, false)
    elseif initialization == :best_rand  # first medoid is random
        indicies = _best_initialization(distances, k, true)
    else
        error("Unknown initialization")
    end

    return _fuzzy_kmedoids_body(distances, k, mval, indicies, max_iter)
end

function _fuzzy_kmedoids_body(distances::Matrix{F}, k_medoids_n::Int, mval::Int, medoids::Vector{Int}, max_iter::Int) :: Vector{Int} where {F<:AbstractFloat}  # Indicies
    old_medoids = zeros(Int, length(medoids))
    new_medoids = copy(medoids)
    mval_other_raise_to = F(1 / mval + 1)
    points_n = size(distances, 1)
    membership_special_final = Matrix{F}(undef, points_n, k_medoids_n)
    objective_matrix = Matrix{F}(undef, points_n, k_medoids_n)
    epsilon = F(1e-8)

    for iter in 1:max_iter
        if old_medoids == new_medoids
            break
        end

        # ------------------------------
        # getting u_ij ^ m form

        # here we calculate normal membership of every point to every medoid
        for (medoid_id, medoid_col) in enumerate(eachcol(membership_special_final))
            medoid_point_id = new_medoids[medoid_id]
            LoopVectorization.@turbo for point_id in 1:points_n
                medoid_col[point_id] = one(F) / (distances[point_id, medoid_point_id] + epsilon) ^ mval
            end
        end
        # We divide it by sum, so that membership sum to 1, membership of each point is rowwise
        for medoid_row in eachrow(membership_special_final)
            medoid_row ./= sum(medoid_row)
        end
        # Up to this point, we have calculated normal membership
        # We apply formula u_ij^m, from paper
        membership_special_final .^= mval_other_raise_to
        # ------------------------------

        # ------------------------------
        # getting new medoids
        old_medoids .= new_medoids

        # It turns out this formula can be represented as atrix multiplication
        LinearAlgebra.mul!(objective_matrix, distances, membership_special_final)
        for (medoid_id, objective_col) in enumerate(eachcol(objective_matrix))
            new_medoids[medoid_id] = argmin(objective_col)
        end
    end

    return new_medoids
end

function _random_initialization(distances::Matrix{F}, k::Int) :: Vector{Int} where {F<:AbstractFloat}
    return Random.randperm(size(distances, 1))[1:k]
end

function _best_initialization(distances::Matrix{F}, k::Int, first_random::Bool) :: Vector{Int} where {F<:AbstractFloat}
    indicies_medoids = zeros(Int, k)
    points_n = size(distances, 1)
    distance_to_closest_medoid = Vector{F}(undef, points_n)
    distance_to_closest_medoid .= typemax(F)

    if first_random
        first_ind = rand(1:points_n)
        indicies_medoids[1] = first_ind
    else
        min_value = typemax(F)
        argmin_candidate_point_id = -1
        for (candidate_point_id, distance_col) in enumerate(eachcol(distances))
            sum_col = sum(distance_col)
            if sum_col < min_value
                min_value = sum_col
                argmin_candidate_point_id = candidate_point_id
            end
        end
        indicies_medoids[1] = argmin_candidate_point_id
    end

    medoid_point_id = indicies_medoids[1]

    # This implements q = arg max[1≤i≤n;i∉V] min[1≤k≤|V|] r(vk, xi)
    for medoid_id in 2:k
        @inbounds @simd for point_id in 1:points_n
            distance_to_closest_medoid[point_id] = min(distance_to_closest_medoid[point_id], distances[point_id, medoid_point_id])
        end
        medoid_point_id = argmax(distance_to_closest_medoid)
        indicies_medoids[medoid_id] = medoid_point_id
    end

    return indicies_medoids
end

end # module FCM