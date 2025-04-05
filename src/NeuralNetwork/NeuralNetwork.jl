
module NeuralNetwork

import Statistics
import Lux
import Random
import IterTools
import Optimisers
import Distances
import LinearAlgebra
import StatsBase
import Logging
import ConcreteStructs
using SimpleChains
using LoopVectorization
using Tullio
# import SimilaritySearch

export AbstractNeuralNetwork, predict, learn!, copy, get_neural_network, get_lux_representation, get_loss, copy_parameters, set_parameters!, copy_state, set_state!

# concrete implementation should have fist parametric type to be a number of dimensions
abstract type AbstractNeuralNetwork end

function get_neural_network(name::Val{T}) where T
    throw("not implemented")
end

function predict(nn::AbstractNeuralNetwork, X) :: Array{Float32}
    throw("not implemented")
end

function get_lux_representation(nn::AbstractNeuralNetwork)
    throw("not implemented")
end

function get_loss(nn::AbstractNeuralNetwork) :: Function
    throw("not implemented")
end

function copy(nn::AbstractNeuralNetwork) :: AbstractNeuralNetwork
    throw("not implemented")
end

function copy_parameters(nn::AbstractNeuralNetwork)
    throw("not implemented")
end

function set_parameters!(nn::AbstractNeuralNetwork, params)
    throw("not implemented")
end

function set_state!(nn::AbstractNeuralNetwork, state)
    throw("not implemented")
end

function copy_state(nn::AbstractNeuralNetwork)
    throw("not implemented")
end

# -------------------------------------------------------
# end interface

# Dummy Neural Network, used for manual testing in visual environments
export DummyNN
struct DummyNN <: AbstractNeuralNetwork end
# -------------------------------------------------------
# import concrete implementations

# general things, important for further imports
include("_ASSEQ.jl")
include("_utils.jl")

# other neural networks
include("_MLP.jl")
include("_DistanceBasedClassificator.jl")
include("_Autoencoder.jl")
include("_Random.jl")
include("_ExemplarBasedNN.jl")

# -------------------------------------------------------
# module functions:

"""
Use this function:
"""
function get_neural_network(name::Symbol)
    return get_neural_network(Val(name))
end

end # module