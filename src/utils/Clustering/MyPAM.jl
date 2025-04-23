module MyPAM

import Distances
import Random
import StatsBase

# actually it is fasterpam
# https://arxiv.org/pdf/1810.05691  -> fastpam1 and fastpam2, I do not use it here
# https://arxiv.org/pdf/2008.05171  -> it is actually faster version, I use it here
# I used lab from first article and swap has some differences to the one from the second article
# My implementation is based on Java ELKI implementation
# https://github.com/elki-project/elki/tree/master/elki-clustering/src/main/java/elki/clustering/kmedoids
export my_pam

struct Cache{F<:AbstractFloat,D<:Distances.PreMetric}
    data::Matrix{F}
    distance::D
    cache::Dict{Int,Vector{F}}
end

function get_distance_vector!(cache::Cache{F}, point_id::Int)::Vector{F} where {F<:AbstractFloat}
    dist_vec = get!(cache.cache, point_id) do
        Distances.colwise(cache.distance, cache.data, view(cache.data, :, point_id))
    end
    return dist_vec
end

function random_initialization(cache::Cache, number_of_medoids::Int)::Vector{Int}
    number_of_points = size(cache.data, 2)
    return Random.randperm(number_of_points)[1:number_of_medoids]
end

@inline function safe_max(::Type{F})::F where {F<:AbstractFloat}
    return prevfloat(typemax(F)) / 2
end

subsamble_basic_size(n::Int, k::Int) = min(10 + ceil(Int, sqrt(n)) * ceil(Int, sqrt(k)), n)
function lab_initialization(
    cache::Cache{F},
    number_of_medoids::Int;
    subsample_size=subsamble_basic_size
)::Vector{Int} where {F<:AbstractFloat}
    number_of_points = size(cache.data, 2)
    medoids = Vector{Int}(undef, number_of_medoids)  # stores the chosen medoid indices

    # Track the distance to the nearest chosen medoid for each point
    distance_to_nearest_medoid = fill(safe_max(F), number_of_points)
    sample_size = subsample_size(number_of_points, number_of_medoids)
    candidate_indices = Random.randperm(number_of_points)[1:sample_size]

    @inbounds @fastmath for current_medoid_index in 1:number_of_medoids
        best_change_in_total_deviation = safe_max(F)
        best_candidate = 0

        # Evaluate each candidate in the subsample to find the one that reduces TD most
        for candidate_point in candidate_indices
            candidate_distances = get_distance_vector!(cache, candidate_point)
            total_deviation_change = zero(F)
            # Compute improvement for this candidate
            @simd for other_point in candidate_indices
                if other_point != candidate_point
                    delta = candidate_distances[other_point] - distance_to_nearest_medoid[other_point]
                    total_deviation_change += ifelse(delta > 0, delta, zero(F))
                end
            end

            if total_deviation_change < best_change_in_total_deviation
                best_change_in_total_deviation = total_deviation_change
                best_candidate = candidate_point
            end
        end

        # Add the chosen candidate as a medoid
        medoids[current_medoid_index] = best_candidate

        best_candidate_matrix = get_distance_vector!(cache, best_candidate)

        # Update distance_to_nearest_medoid for all points
        @simd for point_index in 1:number_of_points
            dist = best_candidate_matrix[point_index]
            distance_to_nearest_medoid[point_index] = ifelse(dist < distance_to_nearest_medoid[point_index], dist, distance_to_nearest_medoid[point_index])
        end
    end

    return medoids
end


function get_point_nearest_second(medoid_distances::Vector{Vector{F}}, medoids::Vector{Int}, point::Int)::Tuple{Int,Int,F,F} where {F<:AbstractFloat}
    best_dist = safe_max(F)
    second_best_dist = safe_max(F)
    best_id = 0
    second_best_id = 0

    @inbounds @fastmath for medoid_id in eachindex(medoids)
        dist = medoid_distances[medoid_id][point]
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

function random_candidates!(;candidates::Vector{Int}, cache::Cache{F},
        dnearest::Vector{F}, nearest_medoid_id::Vector{Int},
        medoids::Vector{Int}, medoid_distances::Vector{Vector{F}}
    )::Vector{Int} where {F<:AbstractFloat}
    number_of_points = size(cache.data, 2)
    candidates .= Random.randperm(number_of_points)[1:length(candidates)]
    return candidates
end

function random_increasing!(;candidates::Vector{Int}, cache::Cache{F},
        dnearest::Vector{F}, nearest_medoid_id::Vector{Int},
        medoids::Vector{Int}, medoid_distances::Vector{Vector{F}}
    )::Vector{Int} where {F<:AbstractFloat}
    number_of_points = size(cache.data, 2)
    increse_number = ceil(Int, sqrt(number_of_points))
    if increse_number > number_of_points - length(candidates)
        increse_number = number_of_points - length(candidates)
    end
    if increse_number <= 0
        return candidates
    end
    new_candidates = zeros(Int, increse_number+length(candidates))
    new_candidates[1:length(candidates)] .= candidates
    allowed = trues(number_of_points)
    allowed[candidates] .= false
    new_candidates[(length(candidates)+1):end] .= Random.shuffle!(collect(filter(x -> allowed[x], 1:number_of_points)))[1:increse_number]
    return new_candidates
end

# It is very weak, so I comment it out
# function closest_candidates!(;candidates::Vector{Int}, cache::Cache{F},
#         dnearest::Vector{F}, nearest_medoid_id::Vector{Int},
#         medoids::Vector{Int}, medoid_distances::Vector{Vector{F}}
#     )::Vector{Int} where {F<:AbstractFloat}
#     number_of_points = size(cache.data, 2)
#     per_each_medoid = ceil(Int, length(candidates) / length(medoids))
#     start = 1
#     finish = per_each_medoid
#     candidates .= 0

#     @inbounds @fastmath for medoid_id in eachindex(medoids)
#         if start > length(candidates)
#             start = length(candidates)
#         end
#         if finish > length(candidates)
#             finish = length(candidates)
#         end

#         max_dist = safe_max(F)

#         for point_id in 1:number_of_points
#             dist = medoid_distances[medoid_id][point_id]

#             if dist < max_dist && dist > zero(F)
#                 max_dist = zero(F)
#                 second_max = zero(F)
#                 max_idx = start
#                 for id in start:finish
#                     dist_tmp = candidates[id] == 0 ? safe_max(F) : medoid_distances[medoid_id][candidates[id]]
#                     if dist_tmp > max_dist
#                         second_max = max_dist
#                         max_dist = dist_tmp
#                         max_idx = id
#                     elseif dist_tmp > second_max
#                         second_max = dist_tmp
#                     end
#                 end

#                 candidates[max_idx] = point_id
#                 max_dist = max(dist, second_max)
#             end
#         end
#         start += per_each_medoid
#         finish += per_each_medoid
#     end

#     return candidates
# end

function weighted_random_candidates!(;candidates::Vector{Int}, cache::Cache{F},
        dnearest::Vector{F}, nearest_medoid_id::Vector{Int},
        medoids::Vector{Int}, medoid_distances::Vector{Vector{F}}
    )::Vector{Int} where {F<:AbstractFloat}
    number_of_points = size(cache.data, 2)

    # Use dnearest directly as weights (already contains distances to nearest medoid)
    weights = copy(dnearest)
    weights[medoids] .= zero(F)  # Exclude current medoids from candidates

    # If all weights are zero, fall back to random selection of non-medoids
    if sum(weights) <= eps(F)
        return random_candidates!(candidates=candidates, cache=cache, dnearest=dnearest, nearest_medoid_id=nearest_medoid_id, medoids=medoids, medoid_distances=medoid_distances)
    end

    # Normalize weights to create a probability distribution
    statsbase_weights = StatsBase.ProbabilityWeights(weights)
    candidates .= StatsBase.sample(1:number_of_points, statsbase_weights, length(candidates), replace=false)

    return candidates
end

function eager_swap_pam(cache::Cache{F}, medoids_initial::Vector{Int}, initial_candidates::Vector{Int}, candidate_function!::Function, max_iter::Int)::Vector{Int} where {F<:AbstractFloat}
    medoids = [medoid for medoid in medoids_initial]
    medoids_n = length(medoids)
    points_n = size(cache.data, 2)
    is_medoid = falses(points_n)
    is_medoid[medoids] .= true
    medoid_distances = [get_distance_vector!(cache, medoid) for medoid in medoids]

    nearest_medoid_id = fill(0, points_n)          # which medoid index (1..k) is nearest
    second_nearest_medoid_id = fill(0, points_n)   # which medoid index (1..k) is second-nearest
    dnearest = fill(safe_max(F), points_n)        # distance to nearest medoid
    dsecond = fill(safe_max(F), points_n)        # distance to second-nearest medoid
    removal_loss = zeros(F, medoids_n)
    candidates = copy(initial_candidates)

    @inbounds for point in 1:points_n
        best_index, second_best_index, best_dist, second_best_dist = get_point_nearest_second(medoid_distances, medoids, point)
        dnearest[point] = best_dist
        dsecond[point] = second_best_dist
        nearest_medoid_id[point] = best_index
        second_nearest_medoid_id[point] = second_best_index
    end

    @inbounds @fastmath @simd for point in 1:points_n
        removal_loss[nearest_medoid_id[point]] += dsecond[point] - dnearest[point]
    end

    deltaTD = Vector{F}(undef, medoids_n)
    previous_medoids = zeros(Int, medoids_n)
    
    for iter in 1:max_iter
        if previous_medoids == medoids
            break
        else
            previous_medoids .= medoids
        end

        candidates = candidate_function!(candidates=candidates, cache=cache, dnearest=dnearest, nearest_medoid_id=nearest_medoid_id, medoids=medoids, medoid_distances=medoid_distances)

        for point_candidate in candidates
            if !is_medoid[point_candidate]
                @inbounds @simd for medoid_id in 1:medoids_n
                    deltaTD[medoid_id] = removal_loss[medoid_id]
                end

                accumulator_for_point_candidate = zero(F)

                candidate_vector = get_distance_vector!(cache, point_candidate)
                @inbounds @fastmath @simd for other_point in 1:points_n
                    if other_point != point_candidate
                        dist_point_candidate = candidate_vector[other_point]
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
                    old_medoid_point = medoids[medoid_id_min]
                    new_medoid_point = point_candidate

                    medoid_distances[medoid_id_min] = candidate_vector
                    is_medoid[old_medoid_point] = false
                    is_medoid[new_medoid_point] = true
                    medoids[medoid_id_min] = new_medoid_point

                    @inbounds @fastmath for other_point in 1:points_n
                        if other_point != new_medoid_point
                            nearest_med_id = nearest_medoid_id[other_point]
                            second_nearest_med_id = second_nearest_medoid_id[other_point]

                            if nearest_med_id == medoid_id_min || second_nearest_med_id == medoid_id_min
                                best_index, second_best_index, best_dist, second_best_dist = get_point_nearest_second(medoid_distances, medoids, other_point)
                                dnearest[other_point] = best_dist
                                dsecond[other_point] = second_best_dist
                                nearest_medoid_id[other_point] = best_index
                                second_nearest_medoid_id[other_point] = second_best_index
                            else
                                # check if new candidate is better than current second
                                dist_new_medoid = candidate_vector[other_point]
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
                    @inbounds @fastmath @simd for point in 1:points_n
                        removal_loss[nearest_medoid_id[point]] += dsecond[point] - dnearest[point]
                    end
                end
            end
        end
    end

    return medoids
end




function my_pam(data::Matrix{F}, distance::Distances.PreMetric, number_of_medoids::Int; max_iter::Int=100, build_method::Symbol=:lab, candidates_method=:random)::Vector{Int} where {F<:AbstractFloat}
    cache = Cache(data, distance, Dict{Int,Vector{F}}())

    if build_method == :random
        initial_medoids = random_initialization(cache, number_of_medoids)
    elseif build_method == :lab
        initial_medoids = lab_initialization(cache, number_of_medoids)
    else
        throw(ArgumentError("Unknown build method: $build_method"))
    end

    candidates_number = subsamble_basic_size(size(data, 2), number_of_medoids)

    if candidates_method == :random
        candidates_function = random_candidates!
    elseif candidates_method == :random_increasing
        candidates_function = random_increasing!
    elseif candidates_method == :weighted_random
        candidates_function = weighted_random_candidates!
    # elseif candidates_method == :best  # THis one doesnt work!
    #     candidates_function = closest_candidates!
    else
        throw(ArgumentError("Unknown candidates method: $candidates_method"))
    end

    initial_candidates = Random.randperm(size(data, 2))[1:candidates_number]
    for (i, candidate) in enumerate(keys(cache.cache))
        if i > length(initial_candidates)
            break
        end
        initial_candidates[i] = candidate
    end
    final_medoids = eager_swap_pam(cache, initial_medoids, initial_candidates, candidates_function, max_iter)
    return final_medoids
end

end