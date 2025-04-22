export MLP_NN

"""
THis is very specific implementation designed for fast inference.
It uses SimpleChains.jl for inference and outputs lux model for training.

Caching concerns:
- SimpleChains.jl uses a lot of caching, even output is cached. Therefore, to freely manipulate the output, I should add this:
    adaptor = Lux.ToSimpleChainsAdaptor((SimpleChains.static(input_size),), true)  # this true is crucial! it will create new array
- tests suggest that these allocations are somehow thread safe, so I can use them in parallel.
"""

"""
We do not store lux parameters, cause we want to keep minimal memory footprint.
"""
mutable struct MLP_NN{CL, ML, MS, PS, SL, ST, F} <: AbstractTrainableAgentNeuralNetwork
    simple_chain_call_last::CL
    model_lux::ML
    model_simple::MS
    parameters_simple::PS
    state_lux::SL
    state_simple::ST
    loss::F
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
                last_activation_function::Symbol=:none,  # previosly was Union{Symbol, Vector{Tuple{Symbol, Int}}, Function}
                loss::Symbol = :mse)
    activation, in_layer = _get_activation_function(input_activation_function)
    loss_function = _get_loss_function(loss)

    lux_layers = []
    simplechains_layers = []

    input_size_tmp = input_size
    # Hidden layers
    for i in 1:hidden_layers
        if in_layer
            layer = Lux.Dense(input_size_tmp => hidden_neurons, activation)
            push!(lux_layers, layer)
            push!(simplechains_layers, layer)
        else
            throw(ArgumentError("Activation function must be applied elementwise for not the last activation function"))
        end
        activation, in_layer = _get_activation_function(activation_function)
        input_size_tmp = hidden_neurons

        if dropout > 0.0
            push!(lux_layers, Lux.Dropout(dropout))
        end
    end

    activation_last, in_layer = _get_activation_function(last_activation_function)
    simple_chain_call_last = identity
    if in_layer
        layer = Lux.Dense(input_size_tmp => output_size, activation_last)
        push!(lux_layers, layer)
        push!(simplechains_layers, layer)
    else
        layer = Lux.Dense(input_size_tmp => output_size)
        push!(lux_layers, layer, activation_last)
        push!(simplechains_layers, layer)
        simple_chain_call_last = activation_last
    end
    
    model_lux = Lux.Chain(lux_layers...)
    model_simple_to_adapt = Lux.Chain(simplechains_layers...)

    adaptor = Lux.ToSimpleChainsAdaptor((SimpleChains.static(input_size),), true)
    simple_model = adaptor(model_simple_to_adapt)

    ps, st = Lux.setup(Random.default_rng(rand(Int)), model_lux)
    st = Lux.testmode(st)

    ps_simple, st_simple = Lux.setup(Random.default_rng(rand(Int)), simple_model)
    # println("states mlp:")
    # display(st)
    mlp = MLP_NN(
        simple_chain_call_last,
        model_lux,
        simple_model,
        ps_simple,
        st,
        st_simple,
        loss_function
    )
    set_parameters!(mlp, ps)
    return mlp
end

function get_loss(nn::MLP_NN) :: Function
    return nn.loss
end

function get_lux_representation(nn::MLP_NN)
    return nn.model_lux
end


function predict(nn::MLP_NN, X::MatrixASSEQ) :: Matrix{Float32}
    return predict(nn, get_nn_input(X))
end

function predict(nn::MLP_NN, X::Matrix{Float32}) :: Matrix{Float32}
    y, _ = Lux.apply(nn.model_simple, X, nn.parameters_simple, nn.state_simple)
    return nn.simple_chain_call_last(y)

    # ----------------------------------------
    # Test if results are newly allocated
    # y_next, _ = Lux.apply(nn.model_simple, X, nn.parameters_simple, nn.state_simple)
    # y[:, 1] .= 0.0f0
    # display(y)
    # display(y_next)

    # -----------------------------------------
    # Test if calculations are thread safe
    # g = 0.0f0
    # tasks_n = 20
    # inputs = [X .* i for i in 1:tasks_n]
    # parameters = [Lux.setup(Random.default_rng(rand(Int)), nn.model_simple)[1] for _ in 1:tasks_n]
    # outputs_normal = Vector{Any}(undef, tasks_n)
    # for i in 1:tasks_n
    #     for _ in 1:1000
    #         y_local, _ = Lux.apply(nn.model_simple, inputs[i], parameters[i], nn.state_simple)
    #         g = y_local[1, 1]
    #         outputs_normal[i] = y_local
    #     end
    # end
    # println("now threaded")
    # outputs = Vector{Any}(undef, tasks_n)
    # # copies = [(deepcopy(nn.model_simple), deepcopy(inputs[i]), deepcopy(nn.state_simple)) for i in 1:8]
    # # Threads.@threads for i in 1:tasks_n
    # #     # model_copy, input_copy, state_copy = copies[i]
    # #     model_copy, input_copy, state_copy = nn.model_simple, inputs[i], nn.state_simple
    # #     for _ in 1:1000
    # #         y_local, _ = Lux.apply(model_copy, input_copy, nn.parameters_simple, state_copy)
    # #         g = y_local[1, 1]
    # #         outputs[i] = y_local
    # #     end
    # # end
    # tasks = [Threads.@spawn begin
    #     # model_copy, input_copy, state_copy = copies[i]
    #     model_copy, input_copy, state_copy = nn.model_simple, inputs[i], nn.state_simple
    #     for _ in 1:1000
    #         y_local, _ = Lux.apply(model_copy, input_copy, parameters[i], state_copy)
    #         g = y_local[1, 1]
    #         outputs[i] = y_local
    #     end
    # end for i in 1:tasks_n]
    # wait.(tasks)
    # for i in 1:tasks_n
    #     println("\n\n")
    #     display(outputs[i])
    #     display(outputs_normal[i])
    # end
    # throw(g)


    # ----------------------------------------------------
    # Test if params translation works between Lux and SimpleChains
    # copied_params = copy_parameters(nn)
    # y_lux, _ = Lux.apply(nn.model_lux, X, copied_params, nn.state_lux)
    # test similarity of output values
    # Test.@test isapprox(y, y_lux; atol=1e-5, rtol=1e-5)
    # display(y_lux)
    # display(y)
    return y
end

function copy(nn::MLP_NN) :: MLP_NN
    return MLP_NN(
        nn.simple_chain_call_last,
        nn.model_lux,
        nn.model_simple,
        deepcopy(nn.parameters_simple),
        deepcopy(nn.state_lux),
        deepcopy(nn.state_simple),
        nn.loss
    )
end

function copy_parameters(nn::MLP_NN)
    example_parameters, _ = Lux.setup(Random.Xoshiro(0), nn.model_lux)
    _copy_params!(example_parameters, nn.parameters_simple.params)
    return example_parameters
end

function set_parameters!(nn::MLP_NN, params)
    _set_params!(nn.parameters_simple.params, params)
end

function copy_state(nn::MLP_NN)
    return deepcopy(nn.state_lux)
end

function set_state!(nn::MLP_NN, state)
    nn.state_lux = Lux.testmode(state)
end

"""
In general, MLP should support this function because of interface it implements.
But cutrrently I do not use it, so it is not implemented yet.
"""
function learn!(nn::MLP_NN, X::Union{MatrixASSEQ, Matrix{Float32}}, Y::Matrix{Float32})
    throw("not implemented")
end
