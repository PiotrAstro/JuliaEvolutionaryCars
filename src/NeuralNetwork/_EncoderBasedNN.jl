export EncoderBasedNN

"""
DistanceBasedClassificator
encoded exemplars have (number of exemplars, features) shape
translation is a vector of size number of exemplars
"""
struct EncoderBasedNN{N <: Union{AbstractAgentNeuralNetwork, AbstractTrainableAgentNeuralNetwork}, FUNA} <: AbstractAgentNeuralNetwork
    encoder::N
    activation_function!::FUNA
    translation::Matrix{Float32}

    function EncoderBasedNN(
        encoder::N,
        translation::Matrix{Float32},  # latent_size x exemplars
        activation_function::Symbol,
    ) where N
        activation_function! = get_activation_layer_function(activation_function)
        activation_fun_type = typeof(activation_function!)

        return new{N, activation_fun_type}(
            encoder,
            activation_function!,
            translation,
        )
    end
end

# ------------------------------------------------------------------------------------------
# main functions
function predict(nn::EncoderBasedNN, X) :: Matrix{Float32}
    actions_by_observation = predict_pre_activation(nn, X)
    for col in eachcol(actions_by_observation)
        nn.activation_function!(col)
    end
    return actions_by_observation
end

function predict_pre_activation(nn::EncoderBasedNN, X) :: Matrix{Float32}
    encoded_x = predict(nn.encoder, X)
    # it will return actions x batch_size
    return nn.translation' * encoded_x
end

# ------------------------------------------------------------------------------------------
# Other interface functions
function get_neural_network(name::Val{:EncoderBasedNN})
    return EncoderBasedNN
end
