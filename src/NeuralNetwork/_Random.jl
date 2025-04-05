export Random_NN


@kwdef struct Random_NN <: AbstractNeuralNetwork
    actions_n::Int
end

function get_neural_network(name::Val{:Random_NN})
    return Random_NN
end

function predict(nn::Random_NN, X::AbstractStateSequence) :: Matrix{Float32}
    random_m = randn(Float32, nn.actions_n, get_length(X))
    return random_m
end
