export DistanceBasedClassificator

"""
DistanceBasedClassificator
encoded exemplars have (number of exemplars, features) shape
translation is a vector of size number of exemplars
"""
struct DistanceBasedClassificator{N <: AbstractNeuralNetwork} <: AbstractNeuralNetwork
    encoder::N
    encoded_exemplars::Matrix{Float32}  # encoded exemplars are normalized (unit vectors) and transposed for better performance, so it should be clusters_n x features
    translation::Vector{Int}
    actions_number::Int
    fuzzy_logic_of_n_closest::Int
end

function get_loss(nn::DistanceBasedClassificator) :: Function
    return get_loss(nn.encoder)
end

function get_Flux_representation(nn::DistanceBasedClassificator)
    return get_Flux_representation(nn.encoder)
end

function predict(nn::DistanceBasedClassificator, X::Array{Float32}) :: Array{Float32}
    # Speed tests:
    # encoded_x = predict(nn.encoder, X)

    # @time for _ in 1:100000
    #     encoded_x = predict(nn.encoder, X)
    # end

    # @time for _ in 1:100000
        
    # end

    # @time for _ in 1:100000
        
    # end

    # println("")
    # sleep(10)

    # --------------------------------------------------------------------------------------------
    # actual Function

    # mean of n closest actions

    encoded_x = predict(nn.encoder, X)

    # normalize length - make it a unit vector
    encoded_x .*= inv.(sqrt.(sum(abs2, encoded_x; dims=1)))
    similarity = nn.encoded_exemplars * encoded_x
    closest_exemplars = [_get_n_largest_indices(one_col, nn.fuzzy_logic_of_n_closest) for one_col in eachcol(similarity)]
    result_matrix = zeros(Float32, nn.actions_number, size(X, 2))
    @inbounds for (result_col, (closest_exemplars_indicies, closest_exemplars_sim)) in zip(eachcol(result_matrix), closest_exemplars)
        # change similarity to distance and clamp from bottom to avoid division by zero
        dist = max.(1.0 .- closest_exemplars_sim, 1e-10)
        min_dist = minimum(dist)
        for (i, row) in enumerate(nn.translation[closest_exemplars_indicies])
            # my fancy formula to get something like percentage
            result_col[row] += min_dist / dist[i]
        end
        # actually, this sum is currently not needed, cause I take max anyway
        result_col ./= sum(result_col)
    end
    return result_matrix

    # ------------------------------------------------
    # original, closest action version
    # encoded_x = predict(nn.encoder, X)
    # encoded_x .*= inv.(sqrt.(sum(abs2, encoded_x; dims=1)))  # normalize length - make it a unit vector
        
    # similarity = nn.encoded_exemplars * encoded_x
    # result_matrix = zeros(Float32, nn.actions_number, size(X, 2))
    # closest_exemplars = [argmax(one_col) for one_col in eachcol(similarity)]
    # row_indices = nn.translation[closest_exemplars]
    # @inbounds for (i, row_index) in enumerate(row_indices)
    #     result_matrix[row_index, i] = 1.0
    # end

    # return result_matrix

    # --------------------------------------------------------------------------------------------

    # # naive implementation
    # encoded_X = predict(nn.encoder, X)
    # distances = Distances.pairwise(Distances.CosineDist(), encoded_exemplars, encoded_X)
    # closest_exemplars = [argmin(one_col) for one_col in eachcol(distances)]
    # result_matrix = zeros(Float32, nn.actions_number, size(X, 2))
    # @inbounds for i in 1:size(X, 2)
    #     result_matrix[nn.translation[closest_exemplars[i]], i] = 1.0
    # end

    # return result_matrix
end

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