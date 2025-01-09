export Combined_NN

# ODN - Output Dimensions Number
struct Combined_NN{T<:Flux.Chain, F} <: AbstractNeuralNetwork
    layers::T
    loss::F
end

function Combined_NN(layers::Vector{<:AbstractNeuralNetwork}) :: Combined_NN
    layers_new = Flux.Chain([get_Flux_representation(layer) for layer in layers])
    loss = layers[end].loss
    return Combined_NN(layers_new, loss)
end

function get_neural_network(name::Val{:Combined_NN})
    return Combined_NN
    
end

function get_loss(nn::Combined_NN) :: Function
    return nn.loss
end

function get_Flux_representation(nn::Combined_NN)
    return nn.layers
end

function predict(nn::Combined_NN, X::Array{Float32})
    return nn.layers(X)
end

function copy(nn::Combined_NN) :: Combined_NN
    return Combined_NN([copy(layer) for layer in nn.layers], nn.loss)
end