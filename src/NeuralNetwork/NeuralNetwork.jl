
module NeuralNetwork

import Statistics
import Flux
import Random
import IterTools
import Optimisers
import Memoization
import Distances
import LinearAlgebra
import SparseArrays
import StatsBase
import Logging
# import SimilaritySearch

export AbstractNeuralNetwork, predict, learn!, copy, get_neural_network, get_Flux_representation, get_loss

# concrete implementation should have fist parametric type to be a number of dimensions
abstract type AbstractNeuralNetwork{ODN} end

function predict(nn::AbstractNeuralNetwork, X::Array{Float32}) :: Array{Float32}
    throw("not implemented")
end

function get_Flux_representation(nn::AbstractNeuralNetwork)
    throw("not implemented")
end

function get_output_dimensions_number(::AbstractNeuralNetwork{ODN}) :: Int where {ODN}
    return ODN
end

function get_loss(nn::AbstractNeuralNetwork) :: Function
    throw("not implemented")
end

function copy(nn::AbstractNeuralNetwork) :: AbstractNeuralNetwork
    throw("not implemented")
end

# -------------------------------------------------------
# end interface

# Dummy Neural Network, used for manual testing in visual environments
export DummyNN
struct DummyNN <: AbstractNeuralNetwork end

# More or less universal function for learning
function learn!(
    nn::AbstractNeuralNetwork,
    X::Array{Float32},
    Y::Array{Float32};
    epochs::Int = 10,
    batch_size::Int = 256,
    learning_rate::AbstractFloat = 0.003,
    verbose::Bool = false
)
    nn_internal = get_Flux_representation(nn)
    nn_loss = get_loss(nn)

    opt_settings = Optimisers.AdamW(learning_rate)
    opt_state = Flux.setup(opt_settings, nn_internal)
    custom_loss = (m, x, y) -> (nn_loss)(m(x), y)

    # shuffle x and y the same way
    perm = Random.randperm(size(X, 2))
    X = X[:, perm]
    Y = Y[:, perm]

    batches = [
        (
        X[:, i: (i+batch_size-1 <= size(X, 2) ? i+batch_size-1 : end)],
        Y[:, i: (i+batch_size-1 <= size(Y, 2) ? i+batch_size-1 : end)]
        ) for i in 1:batch_size:size(X, 2)
    ]

    for epoch in 1:epochs
        Flux.train!(custom_loss, nn_internal, batches, opt_state)

        # print loss
        if verbose
            Logging.@info "Epoch: $epoch, Loss: $(Statistics.mean(custom_loss(nn_internal, X, Y)))\n"
        end
    end
end

# -------------------------------------------------------
# import concrete implementations
include("_MLP.jl")
include("_Combined.jl")
include("_DistanceBasedClassificator.jl")
include("_Autoencoder.jl")

# -------------------------------------------------------
# module functions:

"Should return Type{T} where T<:AbstractNeuralNetwork, but it is impossible to write like that"
function get_neural_network(name::Symbol) :: Type
    if name == :MLP_NN
        return MLP_NN
    elseif name == :Combined_NN
        return Combined_NN
    elseif name == :DistanceBasedClassificator
        return DistanceBasedClassificator
    elseif name == :DummyNN
        return DummyNN
    elseif name == :Autoencoder
        return Autoencoder
    else
        throw("Neural Network not found")
    end
end

end # module