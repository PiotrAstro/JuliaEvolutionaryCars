
module NeuralNetwork

import Statistics
import Lux
import Random
import Optimisers
import Distances
import LinearAlgebra
import StatsBase
import Logging
using SimpleChains
using LoopVectorization
using Tullio

export AbstractNeuralNetwork, predict, learn!, copy, get_neural_network, get_lux_representation, get_loss, copy_parameters, set_parameters!, copy_state, set_state!

# -------------------------------------------------------
# basic neural network interface
abstract type AbstractNeuralNetwork end

function get_neural_network(name::Val{T}) where T
    throw("not implemented")
end

function predict(nn::AbstractNeuralNetwork, X) :: Array{Float32}
    throw("not implemented")
end

function copy(nn::AbstractNeuralNetwork) :: AbstractNeuralNetwork
    throw("not implemented")
end

# --------------------------------------------------------
# interface for trainable neural networks - they should support Lux.jl
# internaly they might use SimpleChains.jl, but interface is Lux.jl
abstract type AbstractTrainableNeuralNetwork <: AbstractNeuralNetwork end

# shoucnt it have learn! function?

function get_lux_representation(nn::AbstractTrainableNeuralNetwork)
    throw("not implemented")
end

function learn!(nn::AbstractTrainableNeuralNetwork, X, Y; kwargs...)
    throw("not implemented")
end

function get_loss(nn::AbstractTrainableNeuralNetwork) :: Function
    throw("not implemented")
end

function copy_parameters(nn::AbstractTrainableNeuralNetwork)
    throw("not implemented")
end

function set_parameters!(nn::AbstractTrainableNeuralNetwork, params)
    throw("not implemented")
end

function set_state!(nn::AbstractTrainableNeuralNetwork, state)
    throw("not implemented")
end

function copy_state(nn::AbstractTrainableNeuralNetwork)
    throw("not implemented")
end

# ---------------------------------------------------------
# interface for functions usable for environment inference
abstract type AbstractAgentNeuralNetwork <: AbstractNeuralNetwork end

function predict(nn::AbstractAgentNeuralNetwork, ASSEQ) :: Matrix{Float32} where {ASSEQ<:AbstractStateSequence}
    throw("not implemented")
end

# ----------------------------------------------------------
# this one is mostly used as encoder for autoencoder
abstract type AbstractTrainableAgentNeuralNetwork <: Union{AbstractAgentNeuralNetwork, AbstractTrainableNeuralNetwork} end

# -------------------------------------------------------
# end interface

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