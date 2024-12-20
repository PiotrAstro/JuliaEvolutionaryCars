export DistanceBasedClassificator

"""
DistanceBasedClassificator
encoded exemplars have (number of exemplars, features) shape
translation is a vector of size number of exemplars
"""
struct DistanceBasedClassificator{N <: AbstractNeuralNetwork, F1<:Function, F2<:Function} <: AbstractNeuralNetwork
    encoder::N
    encoded_exemplars::Matrix{Float32}
    translation::Vector{Int}
    actions_number::Int
    fuzzy_logic_of_n_closest::Int
    closest_exemplars::F1
    final_dist_prepare::F2
    # distance_metric::Symbol # euclidean or cosine or cityblock

    function DistanceBasedClassificator(
        encoder::N,
        encoded_exemplars::Matrix{Float32},
        translation::Vector{Int},
        actions_number::Int,
        fuzzy_logic_of_n_closest::Int,
        distance_metric::Symbol
    ) where N
        if distance_metric == :cosine
            encoded_exemplars = (encoded_exemplars .* inv.(sqrt.(sum(abs2, encoded_exemplars; dims=1))))'
            closest_exemplars = prepare_closest_exemplars_cosine
            final_dist_prepare = final_dist_prepare_cosine
        elseif distance_metric == :euclidean
            closest_exemplars = prepare_closest_exemplars_euclidean
            final_dist_prepare = final_dist_prepare_euclidean
        elseif distance_metric == :cityblock
            closest_exemplars = prepare_closest_exemplars_city_block
            final_dist_prepare = final_dist_prepare_city_block
        else
            throw(ArgumentError("Unknown distance metric: $distance_metric"))
        end
        F1 = typeof(closest_exemplars)
        F2 = typeof(final_dist_prepare)

        # Call new with all type parameters specified
        new{N, F1, F2}(encoder, encoded_exemplars, translation, actions_number, fuzzy_logic_of_n_closest, closest_exemplars, final_dist_prepare)
    end
end

function get_loss(nn::DistanceBasedClassificator) :: Function
    return get_loss(nn.encoder)
end

function get_Flux_representation(nn::DistanceBasedClassificator)
    return get_Flux_representation(nn.encoder)
end

function predict(nn::DistanceBasedClassificator, X::Array{Float32}) ::Matrix{Float32}
    encoded_x = predict(nn.encoder, X)
    closest_exemplars = nn.closest_exemplars(nn, encoded_x)
    result_matrix = zeros(Float32, nn.actions_number, size(X, 2))
    @inbounds for (result_col, (closest_exemplars_indicies, closest_exemplars_dist)) in zip(eachcol(result_matrix), closest_exemplars)
        dist = nn.final_dist_prepare(closest_exemplars_dist)
        min_dist = minimum(dist)
        for (i, row) in enumerate(nn.translation[closest_exemplars_indicies])
            # my fancy formula to get something like percentage
            result_col[row] += min_dist / dist[i]
        end
        # normalize it to look like percentages of each action
        result_col ./= sum(result_col)
    end
    return result_matrix
end

# ------------------------------------------------------------------------------------------
# functions that will pick closest exemplars

function prepare_closest_exemplars_cosine(nn::DistanceBasedClassificator, encoded_x::Matrix{Float32}) :: Vector{Tuple{Vector{Int}, Vector{Float32}}}
    # normalize length - make it a unit vector
    encoded_x .*= inv.(sqrt.(sum(abs2, encoded_x; dims=1)))
    similarity = nn.encoded_exemplars * encoded_x
    closest_exemplars = [_get_n_largest_indices(one_col, nn.fuzzy_logic_of_n_closest) for one_col in eachcol(similarity)]
    return closest_exemplars
end

function prepare_closest_exemplars_euclidean(nn::DistanceBasedClassificator, encoded_x::Matrix{Float32}) :: Vector{Tuple{Vector{Int}, Vector{Float32}}}
    # distances = Distances.pairwise(Distances.Euclidean(), nn.encoded_exemplars, encoded_x)
    @tullio threads=false distances[i,j] := (nn.encoded_exemplars[k,i] - encoded_x[k,j])^2 |> sqrt;
    closest_exemplars = [_get_n_lowest_indices(one_col, nn.fuzzy_logic_of_n_closest) for one_col in eachcol(distances)]
    return closest_exemplars
end

function prepare_closest_exemplars_city_block(nn::DistanceBasedClassificator, encoded_x::Matrix{Float32}) :: Vector{Tuple{Vector{Int}, Vector{Float32}}}
    # distances = Distances.pairwise(Distances.Cityblock(), nn.encoded_exemplars, encoded_x)
    @tullio threads=false distances[i,j] := abs(nn.encoded_exemplars[k,i] - encoded_x[k,j]);
    closest_exemplars = [_get_n_lowest_indices(one_col, nn.fuzzy_logic_of_n_closest) for one_col in eachcol(distances)]
    return closest_exemplars
end

# ------------------------------------------------------------------------------------------
# functions that will prepare final distances, numerical stability etc.

global const EPSILON::Float32 = Float32(1e-10)

function final_dist_prepare_cosine(distances::AbstractVector{Float32}) :: Vector{Float32}
    return max.(1.0 .- distances, EPSILON)
end

function final_dist_prepare_euclidean(distances::AbstractVector{Float32}) :: Vector{Float32}
    return max.(distances, EPSILON)
end

function final_dist_prepare_city_block(distances::AbstractVector{Float32}) :: Vector{Float32}
    return max.(distances, EPSILON)
end

# ------------------------------------------------------------------------------------------
# additional functions

function _get_n_largest_indices(arr::AbstractVector{Float32}, n::Int) :: Tuple{Vector{Int}, Vector{Float32}}
    result = collect(1:n)
    result_values = arr[result]
    min_value, argmin_value = findmin(result_values)

    @inbounds for i in n+1:length(arr)
        value = arr[i]
        if value > min_value
            result[argmin_value] = i
            result_values[argmin_value] = value
            min_value, argmin_value = findmin(result_values)
        end
    end

    return result, result_values
end

function _get_n_lowest_indices(arr::AbstractVector{Float32}, n::Int) :: Tuple{Vector{Int}, Vector{Float32}}
    result = collect(1:n)
    result_values = arr[result]
    max_value, argmax_value = findmax(result_values)

    @inbounds for i in n+1:length(arr)
        value = arr[i]
        if value < max_value
            result[argmax_value] = i
            result_values[argmax_value] = value
            max_value, argmax_value = findmax(result_values)
        end
    end

    return result, result_values
end

# function euclidean_mine_no_sqrt(states1::Matrix{Float32}, states2::Matrix{Float32}) :: Matrix{Float32}
#     n1, n2 = size(states1, 2), size(states2, 2)
#     features = size(states1, 1)
#     result = Matrix{Float32}(undef, n1, n2)

#     @inbounds for i in 1:n1
#         for j in 1:n2
#             s = 0.0f0
#             @simd for k in 1:features
#                 tmp = states1[k, i] - states2[k, j]
#                 s += tmp * tmp
#             end
#             result[i, j] = s
#         end
#     end
#     return result
# end

# function cityblock_mine(states1::Matrix{Float32}, states2::Matrix{Float32}) :: Matrix{Float32}
#     n1, n2 = size(states1, 2), size(states2, 2)
#     features = size(states1, 1)
#     result = Matrix{Float32}(undef, n1, n2)

#     @inbounds for i in 1:n1
#         for j in 1:n2
#             s = 0.0f0
#             @simd for k in 1:features
#                 s += abs(states1[k, i] - states2[k, j])
#             end
#             result[i, j] = s
#         end
#     end

#     return result
# end



# function predict_cosine(nn::DistanceBasedClassificator, X::Array{Float32}) :: Array{Float32}
#     encoded_x = predict(nn.encoder, X)

#     # normalize length - make it a unit vector
#     encoded_x .*= inv.(sqrt.(sum(abs2, encoded_x; dims=1)))
#     similarity = nn.encoded_exemplars * encoded_x
#     closest_exemplars = [_get_n_largest_indices(one_col, nn.fuzzy_logic_of_n_closest) for one_col in eachcol(similarity)]
#     result_matrix = zeros(Float32, nn.actions_number, size(X, 2))
#     @inbounds for (result_col, (closest_exemplars_indicies, closest_exemplars_sim)) in zip(eachcol(result_matrix), closest_exemplars)
#         # change similarity to distance and clamp from bottom to avoid division by zero
#         dist = max.(1.0 .- closest_exemplars_sim, 1e-10)
#         min_dist = minimum(dist)
#         for (i, row) in enumerate(nn.translation[closest_exemplars_indicies])
#             # my fancy formula to get something like percentage
#             result_col[row] += min_dist / dist[i]
#         end
#         # actually, this sum is currently not needed, cause I take max anyway
#         result_col ./= sum(result_col)
#     end
#     return result_matrix
# end