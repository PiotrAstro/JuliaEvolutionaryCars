module NeuralNetworkFunctions
    include("NeuralNetwork.jl")
    include("MLP.jl")
    using .NeuralNetwork
    using .MLP

    export get_neural_network

    "Should return Type{T} where T<:AbstractNeuralNetwork, but it is impossible to write like that"
    function get_neural_network(name::Symbol) :: Type
        if name == :MLP_NN
            return MLP_NN
        else
            throw("Neural Network not found")
        end
    end
end