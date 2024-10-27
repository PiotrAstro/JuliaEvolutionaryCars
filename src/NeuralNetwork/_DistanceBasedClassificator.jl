export DistanceBasedClassificator

"""
DistanceBasedClassificator
encoded exemplars have (number of exemplars, features) shape
translation is a vector of size number of exemplars
"""
struct DistanceBasedClassificator <: AbstractNeuralNetwork
    encoder::AbstractNeuralNetwork
    encoded_exemplars::Matrix{Float32}
    translation::Vector{Int}
    actions_number::Int
end

# function DistanceBasedClassificator(encoder::AbstractNeuralNetwork, encoded_exemplars::Matrix{Float32}, translation::Vector{Int}, actions_number::Int) :: DistanceBasedClassificator 
#     return DistanceBasedClassificator(encoder, encoded_exemplars, translation, actions_number)
# end

function get_parameters(nn::DistanceBasedClassificator) :: Flux.Params
    return get_parameters(nn.encoder)
end

function set_parameters!(nn::DistanceBasedClassificator, parameters::Flux.Params)
    set_parameters!(nn.encoder, parameters)
end

function get_loss(nn::DistanceBasedClassificator) :: Function
    return get_loss(nn.encoder)
end

function get_Flux_representation(nn::DistanceBasedClassificator)
    return get_Flux_representation(nn.encoder)
end

function predict(nn::DistanceBasedClassificator, X::Array{Float32}) :: Array{Float32}
    encoded_x = predict(nn.encoder, X)
    sums = sum(encoded_x, dims=1)
    normalised_x = encoded_x ./ sums
    
    similarity = nn.encoded_exemplars * normalised_x
    result_matrix = zeros(Float32, nn.actions_number, size(X, 2))
    closest_exemplars = [argmax(one_col) for one_col in eachcol(similarity)]
    row_indices = nn.translation[closest_exemplars]
    @inbounds for (i, row_index) in enumerate(row_indices)
        result_matrix[row_index, i] = 1.0
    end

    return result_matrix

    # Just test!!!
    # # encoded_X = X

    # # check which exemplar is the closest
    # # change it to cosine distances using Distances.jl
    # # distances = Distances.pairwise(Distances.CosineDist(), nn.encoded_exemplars, encoded_X)
    # # just for test again, I should use cosine distance
    # # distances = Distances.pairwise(Distances.Euclidean(), nn.encoded_exemplars, encoded_X) 
    # # display(nn.encoded_exemplars)
    # # distances = hcat([vec(sum((nn.encoded_exemplars .- encoded_x).^2, dims=1)) for encoded_x in eachcol(encoded_X)]...)
    # closest_exemplars = [argmin(one_col) for one_col in eachcol(distances)]
    # result_matrix = zeros(Float32, nn.actions_number, size(X, 2))
    # @inbounds for i in 1:size(X, 2)
    #     result_matrix[nn.translation[closest_exemplars[i]], i] = 1.0
    # end
    # # display(result_matrix)
    # # display(distances)
    # # display(closest_exemplars)
    # # display([distances[closest_exemplars[i], i] for i in 1:size(X, 2)])
    # # display(nn.encoded_exemplars[:, closest_exemplars])
    # # display(encoded_X)
    # # sleep(30)

    # return result_matrix
end