
module MLP
    import Flux
    import Random
    import IterTools
    import Optimisers as Opt
    import Memoization as Mm

    import ..NeuralNetwork: AbstractNeuralNetwork, predict, learn!, get_parameters, set_parameters!

    export MLP_NN

    mutable struct MLP_NN <: AbstractNeuralNetwork
        layers::Flux.Chain
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
                    last_activation_function::Union{Symbol, Vector{Tuple{Symbol, Int}}, Function}=:none)
        layers = []
        activation = _get_activation_function(activation_function)[1]
        
        # Hidden layers
        for i in 1:hidden_layers
            if i == 1
                push!(layers, Flux.Dense(input_size, hidden_neurons, activation))
            else
                push!(layers, Flux.Dense(hidden_neurons, hidden_neurons, activation))
            end

            if dropout > 0.0
                push!(layers, Flux.Dropout(dropout))
            end
        end
        if typeof(last_activation_function) <: Symbol
            activation_last_tmp = _get_activation_function(last_activation_function)
            if activation_last_tmp[2]
                push!(layers, Flux.Dense(hidden_neurons, output_size, activation_last_tmp[1]))
            else
                push!(layers, Flux.Dense(hidden_neurons, output_size))
                push!(layers, activation_last_tmp[1])
            end
        elseif typeof(last_activation_function) <: Vector
            push!(layers, Flux.Dense(hidden_neurons, output_size))

            @assert output_size == sum([num for (_, num) in last_activation_function])
            final_activation = _generate_activation_function(last_activation_function)
            push!(layers, final_activation)
        elseif typeof(last_activation_function) <: Function
            push!(layers, Flux.Dense(hidden_neurons, output_size))
            push!(layers, last_activation_function)
        end
        
        return MLP_NN(Flux.Chain(layers))

    end

    function get_parameters(nn::AbstractNeuralNetwork) :: Flux.Params
        return Flux.params(nn.layers)
    end

    function set_parameters!(nn::AbstractNeuralNetwork, parameters::Flux.Params)
        Flux.loadparams!(nn.layers, parameters)
    end

    function predict(nn::MLP_NN, X::Array{Float32}) :: Array{Float32}
        # return Flux.testmode!(nn.layers(X))
        return nn.layers(X)
    end

    function learn!(nn::MLP_NN,
                    X::Array{Float32},
                    Y::Array{Float32},
                    loss::Function;
                    epochs::Int = 64,
                    batch_size::Int = 1,
                    learning_rate::AbstractFloat = 0.01
                    )
        opt_settings = Opt.AdamW(learning_rate)
        opt_state = Flux.setup(opt_settings, nn.layers)
        custom_loss = (m, x, y) -> (loss)(m(x), y)

        # shuffle x and y the same way
        perm = Random.randperm(size(X, 2))
        X = X[:, perm]
        Y = Y[:, perm]

        batches = [(
                X[:, i: (i+batch_size-1 <= size(X, 2) ? i+batch_size-1 : end)],
                Y[:, i: (i+batch_size-1 <= size(Y, 2) ? i+batch_size-1 : end)]
            ) for i in 1:batch_size:size(X, 2)]

        for epoch in 1:epochs
            # for data in batches
            #     gs = Flux.gradient(x -> Loss(nn.layers(x), y_batch), Flux.params(nn.layers))
            #     opt_state, nn.layers = Opt.update!(opt_state, nn.layers, gs)
            # end
            Flux.train!(custom_loss, nn.layers, batches, opt_state)
        end
    end

    # --------------------------------------------------------------------------------
    # protected functions

    function _generate_activation_function_code(activations::Vector{Tuple{Symbol, Int}}) :: Expr
        # Generate start and end indices based on segment lengths
        splits = [num for (_, num) in activations]
        starts = Vector{Int}(undef, length(splits))
        ends = Vector{Int}(undef, length(splits))
        start_tmp = 1
    
        for i in eachindex(splits)
            starts[i] = start_tmp
            start_tmp += splits[i]
            ends[i] = start_tmp - 1
        end
    
        # Get activation function info
        activations_info = [_get_activation_function(activation) for (activation, _) in activations]
    
        # Construct the final expressions
        final_expressions = [
            dot ? :($(activation).(view(x, $(starts[i]):$(ends[i]), :))) : :($(activation)(view(x, $(starts[i]):$(ends[i]), :)))
            # dot ? :($(activation).(x[$(starts[i]):$(ends[i]), :])) : :($(activation)(x[$(starts[i]):$(ends[i]), :]))
            for (i, (activation, dot)) in enumerate(activations_info)
        ]
    
        # Construct the function code
        code = :(x -> vcat($(final_expressions...)))
    
        return code
    end

    Mm.@memoize Dict function _generate_activation_function(activations::Vector{Tuple{Symbol, Int}}) :: Function  # I might remove Dict - it will be a bit faster, but will be based on === not on ==
        f = eval(_generate_activation_function_code(activations))
        return f

        # IMPORTANT: lines below should be uncommented for world age problem, but they also make zygote not work
        
        # final_activation = (x) -> Base.invokelatest(f, x)
        # return final_activation
    end

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
end