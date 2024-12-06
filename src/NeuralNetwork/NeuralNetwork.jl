
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
# import SimilaritySearch

export AbstractNeuralNetwork, predict, learn!, get_parameters, set_parameters!, copy, get_neural_network, get_Flux_representation, get_loss, get_input_representant_nn
abstract type AbstractNeuralNetwork end

# Dummy Neural Network, used for manual testing in visual environments
export DummyNN
struct DummyNN <: AbstractNeuralNetwork end

function predict(nn::AbstractNeuralNetwork, X::Array{Float32}) :: Array{Float32}
    throw("not implemented")
end

function get_parameters(nn::AbstractNeuralNetwork) :: Flux.Params
    throw("not implemented")
end

function set_parameters!(nn::AbstractNeuralNetwork, parameters::Flux.Params)
    throw("not implemented")
end

function get_Flux_representation(nn::AbstractNeuralNetwork)
    throw("not implemented")
end

function get_loss(nn::AbstractNeuralNetwork) :: Function
    throw("not implemented")
end

function copy(nn::AbstractNeuralNetwork) :: AbstractNeuralNetwork
    throw("not implemented")
end

function get_input_representant_nn(nn::AbstractNeuralNetwork)
    throw("not implemented")
end

# More or less universal function for learning
function learn!(
    nn::AbstractNeuralNetwork,
    X::Array{Float32},
    Y::Array{Float32};
    epochs::Int = 10,
    batch_size::Int = 256,
    learning_rate::AbstractFloat = 0.003,
    verbose::Bool = true
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
        # for data in batches
        #     gs = Flux.gradient(x -> Loss(nn.layers(x), y_batch), Flux.params(nn.layers))
        #     opt_state, nn.layers = Opt.update!(opt_state, nn.layers, gs)
        # end
        Flux.train!(custom_loss, nn_internal, batches, opt_state)

        # print loss
        if verbose
            println("Epoch: $epoch, Loss: $(Statistics.mean(custom_loss(nn_internal, X, Y)))")
        end

        # # print accuracy
        # check if one hot encoding
        # predictions = predict(nn, X)
        # accuracy = sum([argmax(predictions[:, i]) == argmax(Y[:, i]) for i in 1:size(Y, 2)]) / size(Y, 2)
        # println("Epoch: $epoch, Accuracy: $accuracy")
    end

    # display(nn.loss)
    # display(nn.layers)
    # display(X)
    # display(Y)
    # display(predict(nn, X))
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