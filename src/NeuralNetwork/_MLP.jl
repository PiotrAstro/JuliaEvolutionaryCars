export MLP_NN

struct MLP_NN{C<:Flux.Chain, F} <: AbstractNeuralNetwork
    layers::C
    loss::F
end

function MLP_NN(layers::Vector{MLP_NN}) :: MLP_NN
    layers_new = Flux.Chain([layer.layers for layer in layers]...)
    loss = layers[end].loss

    return MLP_NN(layers_new, loss)
end

function get_neural_network(name::Val{:MLP_NN})
    return MLP_NN
end

"""
Creates a Multi-Layer Perceptron (MLP) Neural Network.

# Keyword Arguments
- input_size::Int: Size of the input layer.
- output_size::Int: Size of the output layer.
- hidden_layers::Int: Number of hidden layers [0; ∞).
- hidden_neurons::Int: Number of neurons per hidden layer.
- activation_function::Symbol: Activation function for hidden layers.
- last_activation_function::Union{Symbol, Vector{Tuple{Symbol, Int}}}: Activation function for the output layer.
"""
function MLP_NN(;
                input_size::Int,
                output_size::Int,
                hidden_layers::Int=1,
                hidden_neurons::Int=64,
                dropout::Float64=0.0,
                activation_function::Symbol=:relu,
                input_activation_function::Symbol=:none,
                last_activation_function::Union{Symbol, Function}=:none,  # previosly was Union{Symbol, Vector{Tuple{Symbol, Int}}, Function}
                loss::Symbol = :mse)
    layers = []
    activation = _get_activation_function(activation_function)[1]

    loss_function = _get_loss_function(loss)
    
    if input_activation_function != :none
        push!(layers, _get_activation_function(input_activation_function)[1])
    end

    input_size_tmp = input_size
    # Hidden layers
    for i in 1:hidden_layers
        push!(layers, Flux.Dense(input_size_tmp, hidden_neurons, activation))
        input_size_tmp = hidden_neurons

        if dropout > 0.0
            push!(layers, Flux.Dropout(dropout))
        end
    end
    if typeof(last_activation_function) <: Symbol
        activation_last_tmp = _get_activation_function(last_activation_function)
        if activation_last_tmp[2]
            push!(layers, Flux.Dense(input_size_tmp, output_size, activation_last_tmp[1]))
        else
            push!(layers, Flux.Dense(input_size_tmp, output_size))
            push!(layers, activation_last_tmp[1])
        end
    elseif typeof(last_activation_function) <: Function
        push!(layers, Flux.Dense(hidden_neurons, output_size))
        push!(layers, last_activation_function)
    end
    
    return MLP_NN(Flux.Chain(layers...), loss_function)
end

function get_loss(nn::MLP_NN) :: Function
    return nn.loss
end

function get_Flux_representation(nn::MLP_NN)
    return nn.layers
end

function predict(nn::MLP_NN, X::Matrix{Float32}) :: Matrix{Float32}
    # return Flux.testmode!(nn.layers(X))
    return nn.layers(X)
end

function copy(nn::MLP_NN) :: MLP_NN
    return MLP_NN(deepcopy(nn.layers), nn.loss)
end

function get_parameters(nn::MLP_NN)
    return Flux.params(nn.layers)
end
# --------------------------------------------------------------------------------
# protected functions

# function _generate_activation_function_code(activations::Vector{Tuple{Symbol, Int}}) :: Expr
#     # Generate start and end indices based on segment lengths
#     splits = [num for (_, num) in activations]
#     starts = Vector{Int}(undef, length(splits))
#     ends = Vector{Int}(undef, length(splits))
#     start_tmp = 1

#     for i in eachindex(splits)
#         starts[i] = start_tmp
#         start_tmp += splits[i]
#         ends[i] = start_tmp - 1
#     end

#     # Get activation function info
#     activations_info = [_get_activation_function(activation) for (activation, _) in activations]

#     # Construct the final expressions
#     final_expressions = [
#         dot ? :($(activation).(view(x, $(starts[i]):$(ends[i]), :))) : :($(activation)(view(x, $(starts[i]):$(ends[i]), :)))
#         # dot ? :($(activation).(x[$(starts[i]):$(ends[i]), :])) : :($(activation)(x[$(starts[i]):$(ends[i]), :]))
#         for (i, (activation, dot)) in enumerate(activations_info)
#     ]

#     # Construct the function code
#     code = :(x -> vcat($(final_expressions...)))

#     return code
# end

# Memoization.@memoize Dict function _generate_activation_function(activations::Vector{Tuple{Symbol, Int}}) :: Function  # I might remove Dict - it will be a bit faster, but will be based on === not on ==
#     f = eval(_generate_activation_function_code(activations))
#     return f

#     # IMPORTANT: lines below should be uncommented for world age problem, but they also make zygote not work
    
#     # final_activation = (x) -> Base.invokelatest(f, x)
#     # return final_activation
# end

"""
Get activation function from symbol
    
Returns function and information if it should be applied element-wise(True), and to whole array(False).
"""
function _get_activation_function(name::Symbol)::Tuple{Function, Bool}
    if name == :relu
        return (Flux.relu, true)
    elseif name == :sigmoid
        return (Flux.sigmoid, true)
    elseif name == :tanh
        return (Flux.tanh, true)
    elseif name == :softmax
        return (Flux.softmax, false)
    elseif name == :none
        return (identity, false)
    else
        throw("Activation function not implemented")
    end
end

"""
Get loss function from symbol

    Available values:
    - :crossentropy
    - :mse
    - :mae
"""
function _get_loss_function(name::Symbol)
    if name == :crossentropy
        return Flux.crossentropy
    elseif name == :mse
        return Flux.mse
    elseif name == :mae
        return Flux.mae
    elseif name == :kldivergence
        return Flux.kldivergence
    else
        throw("Loss function not implemented")
    end
end