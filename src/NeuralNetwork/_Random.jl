export Random_NN


struct Random_NN <: AbstractNeuralNetwork
    actions_n::Int
end

function get_neural_network(name::Val{:Random_NN})
    return Random_NN
end

function predict(nn::Random_NN, X::Array{Float32}) :: Matrix{Float32}
    random_m = rand(Float32, nn.actions_n, size(X, 2))
    for col in eachcol(random_m)
        col ./= sum(col)
    end
    return random_m
end
