export Random_NN


struct Random_NN{F} <: AbstractAgentNeuralNetwork
    actions_n::Int
    activation!::F

    function Random_NN(actions_n::Int, activation_function::Symbol=:none)
        activation_function! = get_activation_function(activation_function)

        return new{typeof(activation_function!)}(actions_n, activation_function!)
    end

    function Random_NN(actions_n::Int, activation::F) where F
        return new{F}(actions_n, activation)
    end
end

function get_neural_network(name::Val{:Random_NN})
    return Random_NN
end

function predict(nn::Random_NN, X::AbstractStateSequence) :: Matrix{Float32}
    random_m = randn(Float32, nn.actions_n, get_length(X))
    for col in eachcol(random_m)
        nn.activation!(col)
    end
    return random_m
end

function copy(nn::Random_NN) :: Random_NN
    return Random_NN(nn.actions_n, nn.activation!)
end