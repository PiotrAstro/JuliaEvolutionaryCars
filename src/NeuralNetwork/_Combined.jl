export Combined_NN

struct Combined_NN <: AbstractNeuralNetwork
    layers::Flux.Chain
    loss::Function
end

function Combined_NN(layers::Vector{<:AbstractNeuralNetwork}) :: Combined_NN
    layers_new = Flux.Chain([get_Flux_representation(layer) for layer in layers])
    loss = layers[end].loss
    return Combined_NN(layers_new, loss)
end

function get_parameters(nn::Combined_NN) :: Flux.Params
    return Flux.params(nn.layers)
end

function set_parameters!(nn::Combined_NN, parameters::Flux.Params)
    Flux.loadparams!(nn.layers, parameters)
end

function get_loss(nn::Combined_NN) :: Function
    return nn.loss
end

function get_Flux_representation(nn::Combined_NN)
    return nn.layers
end

function predict(nn::Combined_NN, X::Array{Float32}) :: Array{Float32}
    return nn.layers(X)
end