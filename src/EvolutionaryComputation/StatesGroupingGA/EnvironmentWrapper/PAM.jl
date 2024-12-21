module PAM

# actually it is fasterpam
# https://arxiv.org/pdf/1810.05691  -> fastpam1 and fastpam2, I do not use it here
# https://arxiv.org/pdf/2008.05171  -> it is actually faster version, I use it here
export faster_pam


using Statistics

function random_initialization(distances::Matrix{F}, number_of_medoids::Int) :: Vector{Int} where {F <: AbstractFloat}
    number_of_points = size(distances, 1)
    return rand(1:number_of_points, number_of_medoids)
end

subsamble_basic_size(n::Int) = min(10 + ceil(Int, sqrt(n)), n)
function lab_initialization(distance_matrix::Matrix{F},
                            number_of_medoids::Int;
                            subsample_size = subsamble_basic_size) :: Vector{Int} where {F <: AbstractFloat}
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
function get_point_nearest_second(distances::Matrix{F}, medoids::Vector{Int}, point::Int) :: Tuple{Int, Int, F, F} where {F <: AbstractFloat}
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

function eager_swap_pam(distance_matrix::Matrix{F}, medoids_initial::Vector{Int}, max_iter::Int) :: Vector{Int} where {F <: AbstractFloat}
    medoids = [medoid for medoid in medoids_initial]
    medoids_n = length(medoids)
    points_n = size(distance_matrix, 1)
    is_medoid = falses(points_n)
    is_medoid[medoids] .= true

    nearest_medoid_id = fill(0, points_n)          # which medoid index (1..k) is nearest
    second_nearest_medoid_id = fill(0, points_n)   # which medoid index (1..k) is second-nearest
    dnearest = fill(typemax(F), points_n)        # distance to nearest medoid
    dsecond  = fill(typemax(F), points_n)        # distance to second-nearest medoid
    removal_loss = zeros(F, medoids_n)

    @inbounds for point in 1:points_n
        best_index, second_best_index, best_dist, second_best_dist = get_point_nearest_second(distance_matrix, medoids, point)
        dnearest[point] = best_dist
        dsecond[point]  = second_best_dist
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

                @inbounds for medoid_id in 1:medoids_n
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
                    removal_loss[medoid_id_min] = zero(F)

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
                                    dsecond[other_point]  = old_nearest
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
function faster_pam(distance_matrix::Matrix{F}, number_of_medoids::Int; max_iter::Int=100, build_method::Symbol=:lab) :: Vector{Int} where {F <: AbstractFloat}
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






# subsamble_basic_size(n::Int) = min(10 + ceil(Int, sqrt(n)), n)
# """
#     lab_initialization(distance_matrix, number_of_medoids; subsample_size=...)

# Perform the LAB (Linear Approximate BUILD) initialization:

# - distance_matrix: The precomputed distance matrix of size n×n.
# - number_of_medoids: The number of clusters k to choose.
# - subsample_size: A function that determines the size of the random subsample
#   used when selecting each medoid. By default, it uses 10 + sqrt(n).

# Returns:
# - A vector of medoid indices chosen as initial medoids.
# """
# function lab_initialization(distance_matrix::Matrix{F},
#                             number_of_medoids::Int;
#                             subsample_size = subsamble_basic_size) :: Vector{Int} where {F <: AbstractFloat}
#     number_of_points = size(distance_matrix, 1)
#     medoids = Vector{Int}(undef, number_of_medoids)  # stores the chosen medoid indices

#     # Track the distance to the nearest chosen medoid for each point
#     distance_to_nearest_medoid = fill(typemax(F), number_of_points)
#     sample_size = subsample_size(number_of_points)

#     @inbounds for current_medoid_index in 1:number_of_medoids
#         candidate_indices = rand(1:number_of_points, sample_size)

#         best_change_in_total_deviation = typemax(F)
#         best_candidate = 0

#         # Evaluate each candidate in the subsample to find the one that reduces TD most
#         for candidate_point in candidate_indices
#             total_deviation_change = 0.0
#             # Compute improvement for this candidate
#             for other_point in candidate_indices
#                 if other_point != candidate_point
#                     delta = distance_matrix[other_point, candidate_point] - distance_to_nearest_medoid[other_point]
#                     if delta < 0
#                         total_deviation_change += delta
#                     end
#                 end
#             end

#             if total_deviation_change < best_change_in_total_deviation
#                 best_change_in_total_deviation = total_deviation_change
#                 best_candidate = candidate_point
#             end
#         end

#         # Add the chosen candidate as a medoid
#         medoids[current_medoid_index] = best_candidate

#         # Update distance_to_nearest_medoid for all points
#         for point_index in 1:number_of_points
#             dist = distance_matrix[point_index, best_candidate]
#             if dist < distance_to_nearest_medoid[point_index]
#                 distance_to_nearest_medoid[point_index] = dist
#             end
#         end
#     end

#     return medoids
# end

# """
#     fastpam2_swap(distance_matrix, initial_medoids; greedy=true)

# Perform the FastPAM2 SWAP phase:

# - distance_matrix: The precomputed distance matrix.
# - initial_medoids: A vector of medoid indices obtained from initialization.
# - max_iter: Maximum number of iterations to perform.
# - greedy: If true, performs multiple swaps per iteration (fastest convergence).

# Returns:
# - final_medoids: Updated medoids after SWAP converges..
# """
# function fastpam2_swap(distance_matrix::Matrix{F}, initial_medoids::Vector{Int}, max_iter::Int; greedy::Bool=true) :: Vector{Int} where {F <: AbstractFloat} # , Vector{Int}, F
#     number_of_points = size(distance_matrix, 1)
#     medoids = copy(initial_medoids)
#     number_of_medoids = length(medoids)

#     is_medoid = falses(number_of_points)
#     is_medoid[medoids] .= true

#     # Precompute nearest and second-nearest medoids for all points
#     nearest_medoid_id_for = Vector{Int}(undef, number_of_points)  # id of entry in medoids
#     distance_to_nearest = fill(typemax(F), number_of_points)
#     distance_to_second_nearest = fill(typemax(F), number_of_points)

#     # Compute initial nearest and second nearest medoids for each point
#     @inbounds for point_index in 1:number_of_points
#         best_dist = typemax(F)
#         second_best_dist = typemax(F)
#         best_med_id = 0
#         for i in 1:number_of_medoids
#             dist = distance_matrix[point_index, medoids[i]]  # maybe swapped indices will be better for cache localicty? maybe not?
#             if dist < best_dist
#                 second_best_dist = best_dist
#                 best_dist = dist
#                 best_med_id = i
#             elseif dist < second_best_dist
#                 second_best_dist = dist
#             end
#         end
#         distance_to_nearest[point_index] = best_dist
#         distance_to_second_nearest[point_index] = second_best_dist
#         nearest_medoid_id_for[point_index] = best_med_id
#     end

#     candidate_improvements_delta = Vector{F}(undef, number_of_medoids)  # it is here for better memory management
#     best_swap_improvements = zeros(F, number_of_medoids)
#     best_swap_candidates = zeros(Int, number_of_medoids)
#     @inbounds for _ in 1:max_iter
#         # Store best swaps: For each medoid, what is the best swap?
#         best_swap_improvements .= zero(F)
#         best_swap_candidates .= 0

#         improvement_found = false

#         # Evaluate all non-medoids as potential new medoids
#         for candidate_point in 1:number_of_points
#             if !is_medoid[candidate_point]
#                 # For making candidate_point a medoid, track improvement for each medoid to remove
#                 # Start with baseline: removing candidate_point as if it were a medoid (actually we have none yet)
#                 candidate_improvements_baseline = -distance_to_nearest[candidate_point]
#                 candidate_improvements_delta .= zero(F)

#                 for other_point in 1:number_of_points
#                     if other_point != candidate_point
#                         old_distance = distance_to_nearest[other_point]
#                         old_second_distance = distance_to_second_nearest[other_point]
#                         old_medoid_index = nearest_medoid_id_for[other_point]
#                         distance_candidate = distance_matrix[other_point, candidate_point]

#                         # Update the improvement for removing old_medoid
#                         candidate_improvements_delta[old_medoid_index] += min(distance_candidate, old_second_distance) - old_distance

#                         # If the candidate_point medoid is better than old assignment:
#                         if distance_candidate < old_distance
#                             # Reassignment needed: add (distance_candidate - old_distance) to all ∆TD except old_medoid_index
#                             dist_gain = distance_candidate - old_distance
#                             candidate_improvements_baseline += dist_gain
#                             candidate_improvements_delta[old_medoid_index] -= dist_gain
#                         end
#                     end
#                 end

#                 # Find best medoid to replace for this candidate_point
#                 best_improvement_for_this_candidate, best_medoid_to_replace = findmin(candidate_improvements_delta)

#                 # Check if this improves on what we already found for that medoid
#                 best_improvement_real = best_improvement_for_this_candidate + candidate_improvements_baseline
#                 if (
#                     best_improvement_real < 0 &&
#                         (
#                             best_swap_candidates[best_medoid_to_replace] == 0 ||
#                             best_improvement_real < best_swap_improvements[best_medoid_to_replace]
#                         )
#                     )
#                     best_swap_improvements[best_medoid_to_replace] = best_improvement_real
#                     best_swap_candidates[best_medoid_to_replace] = candidate_point
#                     improvement_found = true
#                 end
#             end
#         end

#         if !improvement_found
#             # No improvement found: we have reached a local optimum
#             break
#         end

#         # Perform as many swaps as possible
#         # Sort medoids by improvement to execute best swaps first

#         # medoid_order = sortperm(best_swap_improvements)
#         medoids_ids_changed = [medoid_idx for medoid_idx in 1:number_of_medoids if best_swap_improvements[medoid_idx] < 0]
#         for medoid_idx in medoids_ids_changed
#             old_medoid = medoids[medoid_idx]
#             new_medoid = best_swap_candidates[medoid_idx]

#             is_medoid[old_medoid] = false
#             is_medoid[new_medoid] = true
#             medoids[medoid_idx] = new_medoid

#             # Update nearest and second nearest efficiently:
#             # First handle removal of old medoid:
#             for point_index in 1:number_of_points
#                 if nearest_medoid_id_for[point_index] == medoid_idx
#                     # Need to find a new assignment for this point
#                     best_dist = typemax(F)
#                     second_dist = typemax(F)
#                     best_med_id = 0
#                     for i in 1:number_of_medoids
#                         dist = distance_matrix[point_index, medoids[i]]
#                         if dist < best_dist
#                             second_dist = best_dist
#                             best_dist = dist
#                             best_med_id = i
#                         elseif dist < second_dist
#                             second_dist = dist
#                         end
#                     end
#                     distance_to_nearest[point_index] = best_dist
#                     distance_to_second_nearest[point_index] = second_dist
#                     nearest_medoid_id_for[point_index] = best_med_id
#                 else
#                     dist_to_new = distance_matrix[point_index, new_medoid]
#                     old_nearest_dist = distance_to_nearest[point_index]

#                     if dist_to_new < old_nearest_dist
#                         distance_to_nearest[point_index] = dist_to_new
#                         distance_to_second_nearest[point_index] = old_nearest_dist
#                         nearest_medoid_id_for[point_index] = medoid_idx
#                     elseif dist_to_new < distance_to_second_nearest[point_index]
#                         distance_to_second_nearest[point_index] = dist_to_new
#                     end
#                 end
#             end
#         end
#     end

#     return medoids
# end




# greedy part:
# if greedy
#     medoid_order = sortperm(best_swap_improvements)
#     for medoid_idx in medoid_order
#         if best_swap_improvements[medoid_idx] >= 0
#             # No more improvements
#             break
#         end
#         old_medoid = medoids[medoid_idx]
#         new_medoid = best_swap_candidates[medoid_idx]

#         is_medoid[old_medoid] = false
#         is_medoid[new_medoid] = true
#         medoids[medoid_idx] = new_medoid

#         # Update nearest and second nearest efficiently:
#         # First handle removal of old medoid:
#         for point_index in 1:number_of_points
#             if nearest_medoid_id_for[point_index] == medoid_idx
#                 # Need to find a new assignment for this point
#                 best_dist = typemax(F)
#                 second_dist = typemax(F)
#                 best_med_id = 0
#                 for i in 1:number_of_medoids
#                     dist = distance_matrix[point_index, medoids[i]]
#                     if dist < best_dist
#                         second_dist = best_dist
#                         best_dist = dist
#                         best_med_id = i
#                     elseif dist < second_dist
#                         second_dist = dist
#                     end
#                 end
#                 distance_to_nearest[point_index] = best_dist
#                 distance_to_second_nearest[point_index] = second_dist
#                 nearest_medoid_id_for[point_index] = best_med_id
#             else
#                 dist_to_new = distance_matrix[point_index, new_medoid]
#                 old_nearest_dist = distance_to_nearest[point_index]

#                 if dist_to_new < old_nearest_dist
#                     distance_to_nearest[point_index] = dist_to_new
#                     distance_to_second_nearest[point_index] = old_nearest_dist
#                     nearest_medoid_id_for[point_index] = medoid_idx
#                 elseif dist_to_new < distance_to_second_nearest[point_index]
#                     distance_to_second_nearest[point_index] = dist_to_new
#                 end
#             end
#         end
#     end
# else
#     # Perform only the single best swap overall
#     medoid_idx = argmin(best_swap_improvements)
#     old_medoid = medoids[medoid_idx]
#     new_medoid = best_swap_candidates[medoid_idx]

#     is_medoid[old_medoid] = false
#     is_medoid[new_medoid] = true
#     medoids[medoid_idx] = new_medoid

#     # Update assignments after single best swap:
#     for point_index in 1:number_of_points
#         if nearest_medoid_id_for[point_index] == medoid_idx
#             # Need to find a new assignment for this point
#             best_dist = typemax(F)
#             second_dist = typemax(F)
#             best_med_id = 0
#             for i in 1:number_of_medoids
#                 dist = distance_matrix[point_index, medoids[i]]
#                 if dist < best_dist
#                     second_dist = best_dist
#                     best_dist = dist
#                     best_med_id = i
#                 elseif dist < second_dist
#                     second_dist = dist
#                 end
#             end
#             distance_to_nearest[point_index] = best_dist
#             distance_to_second_nearest[point_index] = second_dist
#             nearest_medoid_id_for[point_index] = best_med_id
#         end
#         dist_to_new = distance_matrix[point_index, new_medoid]
#         old_nearest_dist = distance_to_nearest[point_index]

#         if dist_to_new < old_nearest_dist
#             distance_to_nearest[point_index] = dist_to_new
#             distance_to_second_nearest[point_index] = old_nearest_dist
#             nearest_medoid_id_for[point_index] = medoid_idx
#         elseif dist_to_new < distance_to_second_nearest[point_index]
#             distance_to_second_nearest[point_index] = dist_to_new
#         end
#     end
# end




end