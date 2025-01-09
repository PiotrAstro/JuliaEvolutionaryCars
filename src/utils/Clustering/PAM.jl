module PAM

# actually it is fasterpam
# https://arxiv.org/pdf/1810.05691  -> fastpam1 and fastpam2, I do not use it here
# https://arxiv.org/pdf/2008.05171  -> it is actually faster version, I use it here
# I used lab from first article and swap has some differences to the one from the second article
# My implementation is based on Java ELKI implementation
# https://github.com/elki-project/elki/tree/master/elki-clustering/src/main/java/elki/clustering/kmedoids
export faster_pam

function random_initialization(distances::Matrix{F}, number_of_medoids::Int)::Vector{Int} where {F<:AbstractFloat}
    number_of_points = size(distances, 1)
    return rand(1:number_of_points, number_of_medoids)
end

subsamble_basic_size(n::Int) = min(10 + ceil(Int, sqrt(n)), n)
function lab_initialization(
        distance_matrix::Matrix{F},
        number_of_medoids::Int;
        subsample_size=subsamble_basic_size
    )::Vector{Int} where {F<:AbstractFloat}
    number_of_points = size(distance_matrix, 1)
    medoids = Vector{Int}(undef, number_of_medoids)  # stores the chosen medoid indices

    # Track the distance to the nearest chosen medoid for each point
    distance_to_nearest_medoid = fill(typemax(F), number_of_points)
    sample_size = subsample_size(number_of_points)

    @inbounds for current_medoid_index in 1:number_of_medoids
        candidate_indices = rand(1:number_of_points, sample_size)

        best_change_in_total_deviation = typemax(F)
        best_candidate = 0

        # Evaluate each candidate in the subsample to find the one that reduces TD most
        for candidate_point in candidate_indices
            total_deviation_change = 0.0
            # Compute improvement for this candidate
            for other_point in candidate_indices
                if other_point != candidate_point
                    delta = distance_matrix[other_point, candidate_point] - distance_to_nearest_medoid[other_point]
                    if delta < 0
                        total_deviation_change += delta
                    end
                end
            end

            if total_deviation_change < best_change_in_total_deviation
                best_change_in_total_deviation = total_deviation_change
                best_candidate = candidate_point
            end
        end

        # Add the chosen candidate as a medoid
        medoids[current_medoid_index] = best_candidate

        # Update distance_to_nearest_medoid for all points
        for point_index in 1:number_of_points
            dist = distance_matrix[point_index, best_candidate]
            if dist < distance_to_nearest_medoid[point_index]
                distance_to_nearest_medoid[point_index] = dist
            end
        end
    end

    return medoids
end

@inline
function get_point_nearest_second(distances::Matrix{F}, medoids::Vector{Int}, point::Int)::Tuple{Int,Int,F,F} where {F<:AbstractFloat}
    best_dist = typemax(F)
    second_best_dist = typemax(F)
    best_id = 0
    second_best_id = 0

    @inbounds for medoid_id in eachindex(medoids)
        dist = distances[point, medoids[medoid_id]]
        if dist < best_dist
            second_best_dist = best_dist
            second_best_id = best_id
            best_dist = dist
            best_id = medoid_id
        elseif dist < second_best_dist
            second_best_dist = dist
            second_best_id = medoid_id
        end
    end

    return best_id, second_best_id, best_dist, second_best_dist
end

function eager_swap_pam(distance_matrix::Matrix{F}, medoids_initial::Vector{Int}, max_iter::Int)::Vector{Int} where {F<:AbstractFloat}
    medoids = [medoid for medoid in medoids_initial]
    medoids_n = length(medoids)
    points_n = size(distance_matrix, 1)
    is_medoid = falses(points_n)
    is_medoid[medoids] .= true

    nearest_medoid_id = fill(0, points_n)          # which medoid index (1..k) is nearest
    second_nearest_medoid_id = fill(0, points_n)   # which medoid index (1..k) is second-nearest
    dnearest = fill(typemax(F), points_n)        # distance to nearest medoid
    dsecond = fill(typemax(F), points_n)        # distance to second-nearest medoid
    removal_loss = zeros(F, medoids_n)

    @inbounds for point in 1:points_n
        best_index, second_best_index, best_dist, second_best_dist = get_point_nearest_second(distance_matrix, medoids, point)
        dnearest[point] = best_dist
        dsecond[point] = second_best_dist
        nearest_medoid_id[point] = best_index
        second_nearest_medoid_id[point] = second_best_index
    end

    @inbounds @simd for point in 1:points_n
        removal_loss[nearest_medoid_id[point]] += dsecond[point] - dnearest[point]
    end

    # The snippet uses xlast to detect no improvements
    xlast = 0  # "invalid"
    deltaTD = Vector{F}(undef, medoids_n)
    should_break_outer_loop = false
    swaps = 0
    prew_swaps = -1

    @inbounds for _ in 1:max_iter
        if should_break_outer_loop
            # We have completed scanning all non-medoids without improvement
            break
        end

        if swaps == prew_swaps
            break
        else
            prew_swaps = swaps
        end

        for point_candidate in 1:points_n
            if !is_medoid[point_candidate]
                if point_candidate == xlast
                    should_break_outer_loop = true
                    break
                end

                @inbounds @simd for medoid_id in 1:medoids_n
                    deltaTD[medoid_id] = removal_loss[medoid_id]
                end

                accumulator_for_point_candidate = zero(F)

                for other_point in 1:points_n
                    if other_point != point_candidate
                        dist_point_candidate = distance_matrix[other_point, point_candidate]
                        old_dist = dnearest[other_point]
                        old_second = dsecond[other_point]
                        medoid_id = nearest_medoid_id[other_point]

                        if dist_point_candidate < old_dist
                            accumulator_for_point_candidate += (dist_point_candidate - old_dist)
                            deltaTD[medoid_id] += (old_dist - old_second)
                        elseif dist_point_candidate < old_second
                            deltaTD[medoid_id] += (dist_point_candidate - old_second)
                        end
                    end
                end

                medoid_id_min = argmin(deltaTD)
                best_cost = accumulator_for_point_candidate + deltaTD[medoid_id_min]

                if best_cost < zero(F)
                    swaps += 1
                    old_medoid_point = medoids[medoid_id_min]
                    new_medoid_point = point_candidate
                    xlast = point_candidate

                    is_medoid[old_medoid_point] = false
                    is_medoid[new_medoid_point] = true
                    medoids[medoid_id_min] = new_medoid_point

                    @inbounds for other_point in 1:points_n
                        if other_point != new_medoid_point
                            nearest_med_id = nearest_medoid_id[other_point]
                            second_nearest_med_id = second_nearest_medoid_id[other_point]

                            if nearest_med_id == medoid_id_min || second_nearest_med_id == medoid_id_min
                                best_index, second_best_index, best_dist, second_best_dist = get_point_nearest_second(distance_matrix, medoids, other_point)
                                dnearest[other_point] = best_dist
                                dsecond[other_point] = second_best_dist
                                nearest_medoid_id[other_point] = best_index
                                second_nearest_medoid_id[other_point] = second_best_index
                            else
                                # check if new candidate is better than current second
                                dist_new_medoid = distance_matrix[other_point, new_medoid_point]
                                old_nearest = dnearest[other_point]
                                old_second = dsecond[other_point]

                                if dist_new_medoid < old_nearest
                                    dnearest[other_point] = dist_new_medoid
                                    dsecond[other_point] = old_nearest
                                    nearest_medoid_id[other_point] = medoid_id_min
                                    second_nearest_medoid_id[other_point] = nearest_med_id
                                elseif dist_new_medoid < old_second
                                    dsecond[other_point] = dist_new_medoid
                                    second_nearest_medoid_id[other_point] = medoid_id_min
                                end
                            end
                        end
                    end
                    removal_loss .= zero(F)
                    @inbounds @simd for point in 1:points_n
                        removal_loss[nearest_medoid_id[point]] += dsecond[point] - dnearest[point]
                    end
                end
            end
        end
    end

    return medoids
end




"""
    fast_pam(distance_matrix, number_of_medoids)

Full FastPAM method combining LAB initialization and FastPAM2 SWAP:

- distance_matrix: precomputed distance matrix (n×n)
- number_of_medoids: number of clusters k
- max_iter: maximum number of iterations for SWAP phase
- build_method: currently :random or :lab (linear approximate BUILD)

Returns:
- final_medoids
"""
function faster_pam(distance_matrix::Matrix{F}, number_of_medoids::Int; max_iter::Int=100, build_method::Symbol=:lab)::Vector{Int} where {F<:AbstractFloat}
    if build_method == :random
        initial_medoids = random_initialization(distance_matrix, number_of_medoids)
    elseif build_method == :lab
        initial_medoids = lab_initialization(distance_matrix, number_of_medoids)
    else
        throw(ArgumentError("Unknown build method: $build_method"))
    end
    final_medoids = eager_swap_pam(distance_matrix, initial_medoids, max_iter)
    return final_medoids
end

end