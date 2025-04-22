
"""
BanditPAM ++
based on the paper: https://arxiv.org/pdf/2310.18844
"""
module BanditPAM

import LinearAlgebra
import Random
import Statistics
import LoopVectorization

export bandit_pam_pp

const HUGE_FLOAT = 1e10

mutable struct BanditState{F<:AbstractFloat, FUN}
    data::Matrix{F}
    index_map::Dict{Int, Int}
    state::Matrix{F}
    cache::Dict{Tuple{Int, Int}, F}
    distance_callable::FUN
    permutation::Vector{Int}
    permutation_idx::Int
end

# Maybe Cache should state everything? should try it
function BanditState(data::Matrix{F}, distance_callable, permutation::AbstractVector{Int}, state_width::Int) where {F<:AbstractFloat}
    n = size(data, 2)
    state_real_width = min(state_width, n)
    state = fill(F(-1.0), state_real_width, n)
    index_map = Dict{Int, Int}()
    for i in 1:state_real_width
        index_map[permutation[i]] = i
    end

    return BanditState(
        data,
        index_map,
        state,
        Dict{Tuple{Int, Int}, F}(),
        distance_callable,
        permutation,
        1
    )
end

function get_distance!(state::BanditState{F}, i::Int, j::Int)::F where {F<:AbstractFloat}
    # maybe I should also check i? 
    # index_j = get(state.index_map, j, -1)

    # if index_j != -1
    #     # Check if the distance is already stated
    #     if state.state[index_j, i] != F(-1.0)
    #         return state.state[index_j, i]
    #     else
    #         # Calculate the distance and state it
    #         dist = state.distance_callable(view(state.data, :, i), view(state.data, :, j))
    #         state.state[index_j, i] = dist
    #         return dist
    #     end
    # end
    # return state.distance_callable(view(state.data, :, i), view(state.data, :, j))

    tuple_key = ifelse(i < j, (i, j), (j, i))
    dist = get!(state.cache, tuple_key) do
        state.distance_callable(view(state.data, :, i), view(state.data, :, j))
    end
    return dist

    # return state.distance_callable(view(state.data, :, i), view(state.data, :, j))
end

function get_next_permuted!(state::BanditState, batch_size::Int)
    # permutation_holder = Vector{Int}(undef, batch_size)
    # taken = 0
    # while taken < batch_size
    #     # Check if the current index is within bounds
    #     if state.permutation_idx > length(state.permutation)
    #         state.permutation_idx = 1
    #     end

    #     left = batch_size - taken
    #     end_index = min(state.permutation_idx + left - 1, length(state.permutation))
    #     taken_now = end_index - state.permutation_idx + 1
    #     permutation_holder[taken + 1:taken + taken_now] .= state.permutation[state.permutation_idx:end_index]
    #     taken += taken_now
    #     state.permutation_idx += taken_now
    # end

    # return permutation_holder

    if state.permutation_idx + batch_size - 1 > length(state.permutation)
        state.permutation_idx = 1
    end

    view_holder = view(state.permutation, state.permutation_idx:state.permutation_idx + batch_size - 1)
    state.permutation_idx += batch_size
    return view_holder
end


# -----------------------------------------------------------------------------
# Bandit PAM ++
# -----------------------------------------------------------------------------


function bandit_pam_pp(
        data::Matrix{F},
        distance_callable,
        n_medoids::Int
        ; 
        max_iter::Int=10, 
        batch_size::Int=100, 
        build_confidence::F=F(-9.20f0), 
        swap_confidence::F=F(-9.20f0),
        state_width::Int=500
    ) :: Vector{Int} where {F<:AbstractFloat}
    
    # Generate permutation for reference points if using it
    n = size(data, 2)
    permutation = Random.randperm(n)
    state = BanditState(data, distance_callable, permutation, state_width)
    
    # Perform BUILD step to initialize medoids
    medoid_indices = build!(state, n_medoids, 
                          batch_size=batch_size, 
                          build_confidence=build_confidence)

    # println("build completed: ", medoid_indices)
    
    # If only one medoid is requested, skip the SWAP step
    if n_medoids > 1
        medoid_indices = swap!(state, medoid_indices,
                               batch_size=batch_size,
                               swap_confidence=swap_confidence,
                               max_iter=max_iter)
    end
    
    return medoid_indices
end

function build!(
    state::BanditState{F}, 
    n_medoids::Int;
    batch_size::Int=1000, 
    build_confidence::F=F(3.0f0)
) where {F<:AbstractFloat}
    N = size(state.data, 2)
    
    # Initialize data structures for build phase
    estimates = zeros(F, N)
    best_distances = fill(F(HUGE_FLOAT), N)
    sigma = zeros(F, N)
    candidates = trues(N)
    num_samples = zeros(F, N)
    exact_mask = falses(N)

    compute_exactly = falses(N)
    conf_bound_delta = zeros(F, N)
    ucbs = fill(F(HUGE_FLOAT), N)
    lcbs = zeros(F, N)

    use_absolute = true
    
    # Placeholder for medoid indices
    medoid_indices = zeros(Int, n_medoids)
    
    # Iterate through selecting medoids
    for k in 1:n_medoids
        # println("k: ", k, F(log(N)))
        # Reset tracking variables for this medoid selection
        candidates .= true
        num_samples .= 0
        exact_mask .= false
        estimates .= 0
        state.permutation_idx = 1
        
        # Estimate standard deviation for sampling
        sigma = build_sigma!(state, best_distances, batch_size, use_absolute)
        
        # Prevent zero sigma values
        _non_zero_fill_minimum!(sigma)
        
        # Iterative sampling and elimination process
        while sum(candidates) > 0
            # println("candidates: ", sum(candidates))
            # Determine which points need exact computation
            @inbounds @fastmath @simd for i in 1:N
                compute_exactly[i] = ((num_samples[i] + batch_size) >= N) != exact_mask[i]
            end
            # compute_exactly = ((num_samples .+ batch_size) .>= N) .& .!exact_mask
            
            if any(compute_exactly)
                targets = findall(compute_exactly)
                
                # Compute exact estimates for targets
                result = build_target!(state, targets, best_distances, batch_size, use_absolute, true)
                
                estimates[targets] .= result
                num_samples[targets] .+= N
                exact_mask[targets] .= true
                candidates[targets] .= false
                ucbs[targets] .= result
                lcbs[targets] .= result
            end
            
            # Break if no more candidates
            sum(candidates) == 0 && break
            
            # Sample candidates
            targets = findall(candidates)
            result = build_target!(state, targets, best_distances, batch_size, use_absolute, false)
            
            # Update running average estimates

            # @inbounds @fastmath for (org, i) in enumerate(targets)
            for (org, i) in enumerate(targets)
                estimates[i] = ((num_samples[i] * estimates[i]) + (result[org] * batch_size)) / (batch_size + num_samples[i])
                num_samples[i] += batch_size
                conf_bound_delta[i] = sigma[i] * sqrt((build_confidence + F(log(N))) / num_samples[i])
                ucbs[i] = estimates[i] + conf_bound_delta[i]
                lcbs[i] = estimates[i] - conf_bound_delta[i]
            end
            min_ucbs = _my_minimum(ucbs)
            @inbounds @fastmath @simd for i in targets
                # Update candidates based on confidence bounds
                candidates[i] = (lcbs[i] <= min_ucbs) && !exact_mask[i]
            end
        end
        
        # Select medoid with minimum lower confidence bound
        medoid_index = argmin(estimates)
        medoid_indices[k] = medoid_index
        
        # Update best distances 
        for i in 1:N
            dist = get_distance!(state, i, medoid_index)
            best_distances[i] = min(best_distances[i], dist)
        end
        use_absolute = false
    end
    
    return medoid_indices
end

function build_sigma!(
    state::BanditState{F}, 
    best_distances::Vector{F}, 
    batch_size::Int,
    use_absolute::Bool=false
)::Vector{F} where {F<:AbstractFloat}
    N = size(state.data, 2)
    
    adjusted_batch_size = min(batch_size, N ÷ 4)
    
    # Sample reference points using permutation
    ref_points = get_next_permuted!(state, adjusted_batch_size)
    
    sigma = zeros(F, N)
    sample = zeros(F, length(ref_points))
    
    # Compute standard deviation of distances for each point
    for i in 1:N
        for (ref_point_idx, ref_point) in enumerate(ref_points)
            dist = get_distance!(state, i, ref_point)
            if use_absolute
                sample[ref_point_idx] = dist
            else
                sample[ref_point_idx] = ifelse(
                    dist < best_distances[ref_point], 
                    best_distances[ref_point] - dist, 
                    zero(F)
                )
            end
        end
        sigma[i] = Statistics.std(sample)
    end
    
    return sigma
end

function build_target!(
    state::BanditState{F}, 
    targets::Vector{Int}, 
    best_distances::Vector{F}, 
    batch_size::Int,
    use_absolute::Bool=false,
    exact::Bool=false
)::Vector{F} where {F<:AbstractFloat}
    N = size(state.data, 2)
    
    # Adjust batch size to ensure it doesn't exceed dataset size
    adjusted_batch_size = min(batch_size, N)
    if exact
        adjusted_batch_size = N
    end
    
    # Sample reference points using permutation
    ref_points = get_next_permuted!(state, adjusted_batch_size)
    
    results = zeros(F, length(targets))
    
    # Compute estimated loss for each target
    for (idx, target) in enumerate(targets)
        total = zero(F)
        for ref_point in ref_points
            dist = get_distance!(state, target, ref_point)
            if use_absolute
                total += dist
            elseif dist < best_distances[ref_point]
                total += dist - best_distances[ref_point]
            end
        end
        results[idx] = total / F(adjusted_batch_size)
    end
    
    return results
end


function swap!(
    state::BanditState{F}, 
    medoid_indices::Vector{Int}, 
    ;
    batch_size::Int=1000, 
    swap_confidence::F=F(5.0f0),
    max_iter::Int=100
) where {F<:AbstractFloat}
    N = size(state.data, 2)
    n_medoids = length(medoid_indices)
    
    # Initial assignments 
    assignments = zeros(Int, N)
    best_distances = fill(F(HUGE_FLOAT), N)
    second_best_distances = fill(F(HUGE_FLOAT), N)
    
    # Compute initial assignments and best/second-best distances
    update_assignments!(state, medoid_indices, assignments, 
                        best_distances, second_best_distances)
    
    # Initialize data structures for swap phase
    estimates = zeros(F, n_medoids, N)
    num_samples = zeros(F, n_medoids, N)
    candidates = trues(n_medoids, N)
    exact_mask = falses(n_medoids, N)
    # ucbs = zeros(F, n_medoids, N)
    ucbs = fill(F(HUGE_FLOAT), n_medoids, N)
    lcbs = zeros(F, n_medoids, N)

    compute_exactly = falses(n_medoids, N)
    
    swap_performed = true
    iter = 0
    
    while swap_performed && iter < max_iter
        # println("iter: ", iter)
        iter += 1

        state.permutation_idx = 1
        # Reset tracking variables for this iteration
        
        # Compute sigma for potential swaps
        sigma = swap_sigma!(state, best_distances, second_best_distances, 
                            assignments, batch_size, n_medoids)

        # Prevent zero sigma values
        _non_zero_fill_minimum!(sigma)
        
        # Reset tracking variables
        candidates .= true
        num_samples .= 0
        exact_mask .= false
        estimates .= 0
        
        # Iterative swap evaluation
        while sum(candidates) > 1
            # Determine which points need exact computation
            for i in eachindex(candidates)
                compute_exactly[i] = ((num_samples[i] + batch_size) >= N) != exact_mask[i]
            end
            
            if any(compute_exactly)
                # Find targets that need exact computation
                targets = [j for j in 1:N if any(view(compute_exactly, :, j))]
                result = swap_target!(state, n_medoids, targets, 
                                        best_distances, second_best_distances, 
                                        assignments, batch_size, true)
                
                estimates[:, targets] .= result
                num_samples[:, targets] .+= N
                exact_mask[:, targets] .= true
                ucbs[:, targets] .= result
                lcbs[:, targets] .= result
                min_ucbs = _my_minimum(ucbs)
                for i in eachindex(candidates)
                    candidates[i] = (lcbs[i] <= min_ucbs) && !exact_mask[i]
                end
            end
            
            # Break if no more candidates
            sum(candidates) == 0 && break
            
            # Sample candidates
            targets = [j for j in 1:N if any(view(candidates, :, j))]
            result = swap_target!(state, n_medoids, targets, 
                                  best_distances, second_best_distances, 
                                  assignments, batch_size, false)

            for (j, t) in enumerate(targets)
                @inbounds @fastmath @simd for i in 1:n_medoids
                    estimates[i, t] = ((num_samples[i, t] * estimates[i, t]) + (result[i, j] * batch_size)) / (batch_size + num_samples[i, t])
                    num_samples[i, t] += batch_size
                    conf_bound_delta = sigma[t] * sqrt((swap_confidence + F(log(N))) / num_samples[i, t])
                    ucbs[i, t] = estimates[i, t] + conf_bound_delta
                    lcbs[i, t] = estimates[i, t] - conf_bound_delta
                end
            end
            
            min_ucbs = _my_minimum(ucbs)
            @inbounds @fastmath @simd for i in eachindex(candidates)
                candidates[i] = (lcbs[i] <= min_ucbs) && !exact_mask[i]
            end
        end
        
        # Find best swap
        best_swap_index = argmin(estimates)
        k = best_swap_index[1]
        n = best_swap_index[2]
        
        # Perform swap if beneficial
        if medoid_indices[k] != n
            medoid_indices[k] = n
            swap_performed = true
            
            # Recompute assignments
            update_assignments!(state, medoid_indices, assignments, 
                                best_distances, second_best_distances)
        else
            swap_performed = false
        end
    end
    
    return medoid_indices
end

function swap_sigma!(
    state::BanditState{F}, 
    best_distances::Vector{F}, 
    second_best_distances::Vector{F},
    assignments::Vector{Int}, 
    batch_size::Int,
    n_medoids::Int=1
)::Matrix{F} where {F<:AbstractFloat}
    N = size(state.data, 2)
    K = n_medoids
    
    # Adjust batch size to ensure it doesn't exceed dataset size
    adjusted_batch_size = min(batch_size, N ÷ 4)
    
    # Sample reference points using permutation
    ref_points = get_next_permuted!(state, adjusted_batch_size)
    
    sigma = zeros(F, K, N)
    sample = zeros(F, length(ref_points))
    
    # Compute standard deviation for each potential swap
    for n in 1:N, k in 1:K
        for (j, ref_point) in enumerate(ref_points)
            dist = get_distance!(state, n, ref_point)
            
            if k == assignments[ref_point]
                sample[j] = ifelse(
                    dist < second_best_distances[ref_point], 
                    dist - best_distances[ref_point], 
                    zero(F)
                )
            else
                sample[j] = ifelse(
                    dist < best_distances[ref_point], 
                    dist - best_distances[ref_point], 
                    zero(F)
                )
            end
        end
        
        sigma[k, n] = Statistics.std(sample)
    end
    
    return sigma
end

function swap_target!(
    state::BanditState{F}, 
    n_medoids::Int,
    targets::Vector{Int}, 
    best_distances::Vector{F}, 
    second_best_distances::Vector{F},
    assignments::Vector{Int}, 
    batch_size::Int,
    exact::Bool=false
)::Matrix{F} where {F<:AbstractFloat}
    N = size(state.data, 2)
    
    # Adjust batch size to ensure it doesn't exceed dataset size
    adjusted_batch_size = exact ? N : min(batch_size, N)
    
    # Sample reference points using permutation
    ref_points = get_next_permuted!(state, adjusted_batch_size)
    
    results = zeros(F, n_medoids, length(targets))
    
    # Compute swap estimates for each target
    for (t_idx, target) in enumerate(targets)
        for ref_point in ref_points
            dist = get_distance!(state, target, ref_point)
            k = assignments[ref_point]
            
            # Update results based on potential improvement
            if dist < best_distances[ref_point]
                results[:, t_idx] .+= dist - best_distances[ref_point]
            end
            
            # Adjust for current medoid assignment
            results[k, t_idx] += 
                min(dist, second_best_distances[ref_point]) - 
                min(dist, best_distances[ref_point])
        end
    end

    results .*= one(F) / F(adjusted_batch_size)
    
    return results
end

function update_assignments!(
    state::BanditState{F}, 
    medoid_indices::Vector{Int}, 
    assignments::Vector{Int},
    best_distances::Vector{F}, 
    second_best_distances::Vector{F}
) where {F<:AbstractFloat}
    N = size(state.data, 2)
    
    # Compute assignments and distances
    for i in 1:N
        for (k, medoid) in enumerate(medoid_indices)
            dist = get_distance!(state, i, medoid)
            
            if dist < best_distances[i]
                second_best_distances[i] = best_distances[i]
                best_distances[i] = dist
                assignments[i] = k
            elseif dist < second_best_distances[i]
                second_best_distances[i] = dist
            end
        end
    end
end

function _my_minimum(arr::AbstractArray)
    @inbounds min_val = arr[1]
    LoopVectorization.@turbo for i in eachindex(arr)
        min_val = ifelse(arr[i] < min_val, arr[i], min_val)
    end
    return min_val
end

function _non_zero_fill_minimum!(arr::AbstractArray{F}) where {F<:AbstractFloat}
    min_val = F(HUGE_FLOAT)
    @inbounds @fastmath @simd for i in eachindex(arr)
        bool_val = arr[i] > zero(F) && arr[i] < min_val
        min_val = ifelse(bool_val, arr[i], min_val)
    end

    LoopVectorization.@turbo for i in eachindex(arr)
        arr[i] = ifelse(arr[i] > zero(F), arr[i], min_val)
    end
end

end # module
