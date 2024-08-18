
module NeuralNetwork
import Flux
import Random
import IterTools
import Optimisers
import Memoization

export AbstractNeuralNetwork, DummyNN, predict, learn!, get_parameters, set_parameters!, get_neural_network
abstract type AbstractNeuralNetwork end

function predict(nn::AbstractNeuralNetwork, X::Array{Float32}) :: Array{Float32}
    throw("not implemented")
end

function learn!(nn::AbstractNeuralNetwork, X::Array{Float32}, Y::Array{Float32}; epochs::Int = 20, batch_size::Int = 1, learning_rate::AbstractFloat = 0.01)
    throw("not implemented")
end

function get_parameters(nn::AbstractNeuralNetwork) :: Flux.Params
    throw("not implemented")
end

function set_parameters!(nn::AbstractNeuralNetwork, parameters::Flux.Params)
    throw("not implemented")
end

# -------------------------------------------------------
# import concrete implementations
include("_MLP.jl")

# -------------------------------------------------------
# module functions:

"Should return Type{T} where T<:AbstractNeuralNetwork, but it is impossible to write like that"
function get_neural_network(name::Symbol) :: Type
    if name == :MLP_NN
        return MLP_NN
    else
        throw("Neural Network not found")
    end
end
end # module