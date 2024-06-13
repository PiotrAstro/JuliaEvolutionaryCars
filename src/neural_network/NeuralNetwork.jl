module NeuralNetwork
    using Flux

    export AbstractNeuralNetwork, predict, learn!, get_parameters, set_parameters!
    abstract type AbstractNeuralNetwork end

    function predict(nn::AbstractNeuralNetwork, X::Array{Float32}) :: Array{Float32}
        throw("not implemented")
    end

    function learn!(nn::AbstractNeuralNetwork, X::Array{Float32}, Y::Array{Float32}, Losses::Function; epochs::Int = 1, batch_size::Int = 1, learning_rate::AbstractFloat = 0.01)
        throw("not implemented")
    end

    function get_parameters(nn::AbstractNeuralNetwork) :: Flux.Params
        throw("not implemented")
    end

    function set_parameters!(nn::AbstractNeuralNetwork, parameters::Flux.Params)
        throw("not implemented")
    end
end # module