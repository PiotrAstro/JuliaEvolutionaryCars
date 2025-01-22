export DistanceBasedClassificator, membership, encoded_membership

"""
DistanceBasedClassificator
encoded exemplars have (number of exemplars, features) shape
translation is a vector of size number of exemplars
"""
struct DistanceBasedClassificator{M<:Val, N <: AbstractNeuralNetwork, F<:Function, F2<:Function} <: AbstractNeuralNetwork
    encoder::N
    encoded_exemplars::Matrix{Float32}
    translation::Matrix{Float32}
    actions_number::Int
    fuzzy_logic_of_n_closest::Int
    distance_function!::F
    predict_function::F2

    function DistanceBasedClassificator(
        encoder::N,
        encoded_exemplars::Matrix{Float32},
        translation::Matrix{Float32},
        actions_number::Int,
        fuzzy_logic_of_n_closest::Int,
        distance_metric::Symbol,
        m_value::Int = 1,  # m value for fuzzy membership function, it is (1 / distance[i]) ^ m_value, usually 2, I used 1
    ) where N
        if distance_metric == :cosine
            encoded_exemplars = collect(normalize_unit(encoded_exemplars)')
            distance_function! = distance_cosine!
        elseif distance_metric == :euclidean
            distance_function! = distance_euclidean
        elseif distance_metric == :cityblock
            distance_function! = distance_cityblock
        else
            throw(ArgumentError("Unknown distance metric: $distance_metric"))
        end

        if m_value < 1
            throw(ArgumentError("m_value should be greater than 0"))
        end

        predict_function = predict_n_closest
        if fuzzy_logic_of_n_closest == size(encoded_exemplars, 1) || fuzzy_logic_of_n_closest < 1
            predict_function = predict_all
        end

        F = typeof(distance_function!)
        F2 = typeof(predict_function)

        # Call new with all type parameters specified
        new{Val{m_value}, N, F, F2}(encoder, encoded_exemplars, translation, actions_number, fuzzy_logic_of_n_closest, distance_function!, predict_function)
    end
end

function DistanceBasedClassificator(
    encoder::N,
    encoded_exemplars::Matrix{Float32},
    translation::Vector{Int},
    actions_number::Int,
    fuzzy_logic_of_n_closest::Int,
    distance_metric::Symbol,
    m_value::Int = 1,  # m value for fuzzy membership function, it is (1 / distance[i]) ^ m_value, usually 2, I used 1
) where N
    translation_new = zeros(Float32, actions_number, length(translation))
    for (i, row) in enumerate(translation)
        translation_new[row, i] = 1.0f0
    end
    return DistanceBasedClassificator(
        encoder,
        encoded_exemplars,
        translation_new,
        actions_number,
        fuzzy_logic_of_n_closest,
        distance_metric,
        m_value
    )
end

function get_neural_network(name::Val{:DistanceBasedClassificator})
    return DistanceBasedClassificator
end

function predict_n_closest(nn::DistanceBasedClassificator{Val{MINT}}, distances::Matrix{Float32})::Matrix{Float32} where {MINT}
    closest_exemplars = [_get_n_lowest_indices(one_col, nn.fuzzy_logic_of_n_closest) for one_col in eachcol(distances)]
    result_matrix = zeros(Float32, nn.actions_number, size(distances, 2))
    @inbounds for (result_col, (closest_exemplars_indicies, closest_exemplars_dist)) in zip(eachcol(result_matrix), closest_exemplars)
        @fastmath for i in eachindex(closest_exemplars_dist)
            member = (1.0f0 / closest_exemplars_dist[i]) ^ MINT
            exemplars_id = closest_exemplars_indicies[i]
            @simd for row in 1:nn.actions_number
                result_col[row] += nn.translation[row, exemplars_id] * member
            end
        end
        result_col ./= sum(result_col)
    end
    return result_matrix
end

function predict_all(nn::DistanceBasedClassificator{Val{M_INT}}, distances::Matrix{Float32})::Matrix{Float32} where {M_INT}
    result_matrix = zeros(Float32, nn.actions_number, size(distances, 2))
    @inbounds for (i, result_col) in enumerate(eachcol(result_matrix))
        LoopVectorization.@turbo for exemplar_id in axes(nn.translation, 2)
            member = (1.0f0 / distances[exemplar_id, i]) ^ M_INT
            for row in 1:nn.actions_number
                result_col[row] += nn.translation[row, exemplar_id] * member
            end
        end
        @fastmath inv_sum = 1.0f0 / sum(result_col)
        @fastmath result_col .*= inv_sum
    end
    return result_matrix
end

global const EPSILON::Float32 = Float32(1e-6)

function get_loss(nn::DistanceBasedClassificator) :: Function
    return get_loss(nn.encoder)
end

function get_Flux_representation(nn::DistanceBasedClassificator)
    return get_Flux_representation(nn.encoder)
end

# using BenchmarkTools
function predict(nn::DistanceBasedClassificator, X)::Matrix{Float32}
    # # encoded_x = predict(nn.encoder, X)
    # # distances = nn.distance_function!(nn.encoded_exemplars, encoded_x)
    # # display(nn.translation)
    # # sleep(2)
    # # display(distances)
    # # sleep(2)
    # # display(membership(nn, X))
    # # sleep(2)
    # # display(predict_n_closest(nn, distances))
    # # sleep(2)
    # # display(predict_all(nn, distances))
    # # sleep(2)

    # b = @benchmark predict(($nn).encoder, $X)
    # display(b)
    # sleep(2)

    # b = @benchmark ($nn).distance_function!(($nn).encoded_exemplars, predict(($nn).encoder, $X))
    # display(b)
    # sleep(2)

    # distances = nn.distance_function!(nn.encoded_exemplars, predict(nn.encoder, X))

    # b = @benchmark predict_all($nn, $distances)
    # display(b)
    # sleep(2)

    # b = @benchmark predict_tmp_test($nn, $X)
    # display(b)
    # sleep(2)

    # println("Predicted\n\n")
    # throw("random hgidfg")

    encoded_x = predict(nn.encoder, X)
    distances = nn.distance_function!(nn.encoded_exemplars, encoded_x)
    return nn.predict_function(nn, distances)
end

# function predict_tmp_test(nn::DistanceBasedClassificator, X)::Matrix{Float32}
#     encoded_x = predict(nn.encoder, X)
#     distances = nn.distance_function!(nn.encoded_exemplars, encoded_x)
#     return nn.predict_function(nn, distances)
# end

function membership(nn::DistanceBasedClassificator, X)::Matrix{Float32}
    encoded_x = predict(nn.encoder, X)
    return encoded_membership(nn, encoded_x)
end

function encoded_membership(nn::DistanceBasedClassificator{Val{MINT}}, encoded_x::Matrix{Float32})::Matrix{Float32} where {MINT}
    encoded_copy = Base.copy(encoded_x)
    distances = nn.distance_function!(nn.encoded_exemplars, encoded_copy)
    membership_matrix = zeros(Float32, size(distances, 1), size(distances, 2))
    @inbounds @fastmath for (i, col) in enumerate(eachcol(membership_matrix))
        # Could be turbo? LoopVectorization.@turbo 
        @simd for row_ind in eachindex(col)
            col[row_ind] = (1.0f0 / distances[row_ind, i]) ^ MINT
        end
        col ./= sum(col)
    end
    return membership_matrix
end

# ------------------------------------------------------------------------------------------

function distance_cosine!(exemplars::Matrix{Float32}, encoded_x::Matrix{Float32}) :: Matrix{Float32}
    normalize_unit!(encoded_x)
    distances = exemplars * encoded_x
    @inbounds @fastmath for i in eachindex(distances)
        val = 1.0f0 - distances[i]
        distances[i] = ifelse(val < EPSILON, EPSILON, val)
    end
    return distances
end

function distance_euclidean(exemplars::Matrix{Float32}, encoded_x::Matrix{Float32}) :: Matrix{Float32}
    # distances = Distances.pairwise(Distances.Euclidean(), nn.encoded_exemplars, encoded_x)
    @tullio threads=false distances[i,j] := (exemplars[k,i] - encoded_x[k,j])^2 |> sqrt;
    @inbounds @simd for i in eachindex(distances)
        distances[i] = ifelse(distances[i] < EPSILON, EPSILON, distances[i])
    end
    return distances
end

function distance_cityblock(exemplars::Matrix{Float32}, encoded_x::Matrix{Float32}) :: Matrix{Float32}
    # distances = Distances.pairwise(Distances.Cityblock(), nn.encoded_exemplars, encoded_x)
    @tullio threads=false distances[i,j] := abs(exemplars[k,i] - encoded_x[k,j]);
    @inbounds @simd for i in eachindex(distances)
        distances[i] = ifelse(distances[i] < EPSILON, EPSILON, distances[i])
    end
    return distances
end

# ------------------------------------------------------------------------------------------

function normalize_unit(x::Matrix{Float32}) :: Matrix{Float32}
    copied = Base.copy(x)
    normalize_unit!(copied)
    return copied
end

function normalize_unit!(x::Matrix{Float32})
    # for col in eachcol(x)
    #     LinearAlgebra.normalize!(col)
    # end
    for col_ind in axes(x, 2)
        sum_squared = 0.0f0
        @turbo for row_ind in axes(x, 1)
            sum_squared += x[row_ind, col_ind] ^ 2
        end
        sum_squared = 1.0f0 / sqrt(sum_squared)
        @turbo for row_ind in axes(x, 1)
            x[row_ind, col_ind] *= sum_squared
        end
    end
end

# ------------------------------------------------------------------------------------------
# additional functions

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









# function predict_n_closest(nn::DistanceBasedClassificator{Val{MINT}}, distances::Matrix{Float32})::Matrix{Float32} where {MINT}
#     closest_exemplars = [_get_n_lowest_indices(one_col, nn.fuzzy_logic_of_n_closest) for one_col in eachcol(distances)]
#     result_matrix = zeros(Float32, nn.actions_number, size(distances, 2))
#     @inbounds for (result_col, (closest_exemplars_indicies, closest_exemplars_dist)) in zip(eachcol(result_matrix), closest_exemplars)
#         for (i, row) in enumerate(@view nn.translation[closest_exemplars_indicies])
#             @fastmath result_col[row] += (1.0f0 / closest_exemplars_dist[i]) ^ MINT
#         end
#         @fastmath result_col ./= sum(result_col)
#     end
#     return result_matrix
# end

# function predict_all(nn::DistanceBasedClassificator{Val{M_INT}}, distances::Matrix{Float32})::Matrix{Float32} where {M_INT}
#     result_matrix = zeros(Float32, nn.actions_number, size(distances, 2))
#     @inbounds for (i, result_col) in enumerate(eachcol(result_matrix))
#         @fastmath @simd for exemplar_id in eachindex(nn.translation)
#             result_col[nn.translation[exemplar_id]] += (1.0f0 / distances[exemplar_id, i]) ^ M_INT
#         end
#         result_col ./= sum(result_col)
#     end
#     return result_matrix
# end


# ------------------------------------------------------------------------------------------
# Concrete continuous based
