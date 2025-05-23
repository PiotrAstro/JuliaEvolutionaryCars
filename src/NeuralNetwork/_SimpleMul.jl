export SimpleMul_NN


struct SimpleMul_NN{F} <: AbstractAgentNeuralNetwork
    mul_mat::Matrix{Float32}
    activation!::F

    function SimpleMul_NN(mul_mat::Matrix{Float32}, activation_function::Symbol=:none)
        activation_function! = get_activation_layer_function(activation_function)

        return new{typeof(activation_function!)}(mul_mat, activation_function!)
    end

    function SimpleMul_NN(mul_mat::Int, activation::F) where F
        return new{F}(mul_mat, activation)
    end
end

function get_neural_network(name::Val{:SimpleMul_NN})
    return SimpleMul_NN
end

function predict(nn::SimpleMul_NN, X::MatrixASSEQ) :: Matrix{Float32}
    input = X.states
    decision = nn.mul_mat' * input
    for col in eachcol(decision)
        nn.activation!(col)
    end
    return decision
end

function copy(nn::SimpleMul_NN) :: SimpleMul_NN
    return SimpleMul_NN(nn.mul_mat, nn.activation!)
end