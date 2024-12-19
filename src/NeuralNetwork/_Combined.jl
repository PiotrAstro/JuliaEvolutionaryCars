export Combined_NN

# ODN - Output Dimensions Number
struct Combined_NN{ODN} <: AbstractNeuralNetwork{ODN}
    layers::Flux.Chain
    output_size::Int
    loss::Function
end

function Combined_NN(layers::Vector{<:AbstractNeuralNetwork}) :: Combined_NN
    layers_new = Flux.Chain([get_Flux_representation(layer) for layer in layers])
    output_size = get_output_dimensions_number(layers[end])
    loss = layers[end].loss
    return Combined_NN{output_size}(layers_new, loss)
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

function copy(nn::Combined_NN{ODN}) :: Combined_NN where {ODN}
    return Combined_NN{ODN}([copy(layer) for layer in nn.layers], nn.loss)
end