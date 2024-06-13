module MLP
    include("NeuralNetwork.jl")
    using .NeuralNetwork
    using Flux
    using Memoization

    export MLP_NN, predict, learn!, get_parameters, set_parameters!

    struct MLP_NN <: NeuralNetwork.AbstractNeuralNetwork
        layers::Chain
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
                    #last_activation_function::Union{Symbol, Vector{Tuple{Symbol, Int}}}=:none)
                    last_activation_function::Union{Symbol, Vector{Tuple{Symbol, Int}}, Function}=:none)
        layers = []
        activation = _get_activation_function(activation_function)[1]
        
        # Hidden layers
        for i in 1:hidden_layers
            if i == 1
                push!(layers, Dense(input_size, hidden_neurons, activation))
            else
                push!(layers, Dense(hidden_neurons, hidden_neurons, activation))
            end

            if dropout > 0.0
                push!(layers, Dropout(dropout))
            end
        end
        if typeof(last_activation_function) <: Symbol
            activation_last_tmp = _get_activation_function(last_activation_function)
            if activation_last_tmp[2]
                push!(layers, Dense(hidden_neurons, output_size, activation_last_tmp[1]))
            else
                push!(layers, Dense(hidden_neurons, output_size))
                push!(layers, activation_last_tmp[1])
            end
        elseif typeof(last_activation_function) <: Vector
            push!(layers, Dense(hidden_neurons, output_size))

            @assert output_size == sum([num for (_, num) in last_activation_function])
            final_activation = _generate_activation_function(last_activation_function)
            push!(layers, final_activation)
        elseif typeof(last_activation_function) <: Function
            push!(layers, Dense(hidden_neurons, output_size))
            push!(layers, last_activation_function)
        end
        
        return MLP_NN(Chain(layers))

    end

    function get_parameters(nn::AbstractNeuralNetwork) :: Flux.Params
        return Flux.params(nn.layers)
    end

    function set_parameters!(nn::AbstractNeuralNetwork, parameters::Flux.Params)
        Flux.loadparams!(nn.layers, parameters)
    end

    function predict(nn::MLP_NN, X::Array{Float32}) :: Array{Float32}
        return testmode!(nn.layers(X))
    end

    function learn!(nn::MLP_NN,
                    X::Array{Float32},
                    Y::Array{Float32},
                    Loss::Function;
                    epochs::Int = 1,
                    batch_size::Int = 1,
                    learning_rate::AbstractFloat = 0.01
                    )
        data = [(X, Y)]
        opt = ADAM(learning_rate)
        ps = Flux.params(nn.layers)
        for epoch in 1:epochs
            Flux.train!(Losses, ps, data, opt)
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
            dot ? :($(activation).(x[$(starts[i]):$(ends[i]), :])) : :($(activation)(x[$(starts[i]):$(ends[i]), :]))
            for (i, (activation, dot)) in enumerate(activations_info)
        ]
    
        # Construct the function code
        code = :(x -> vcat($(final_expressions...)))
    
        return code
    end

    @memoize Dict function _generate_activation_function(activations::Vector{Tuple{Symbol, Int}}) :: Function  # I might remove Dict - it will be a bit faster, but will be based on === not on ==
        f = eval(_generate_activation_function_code(activations))
        final_activation = (x) -> Base.invokelatest(f, x)
        return final_activation
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