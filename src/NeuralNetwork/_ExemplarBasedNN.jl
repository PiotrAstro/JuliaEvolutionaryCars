export DistanceBasedClassificator, membership, encoded_membership, get_states_number

global const EPSILON_EXEMPLARBASEDNN::Float32 = Float32(1e-6)

"""
DistanceBasedClassificator
encoded exemplars have (number of exemplars, features) shape
translation is a vector of size number of exemplars
"""
struct ExemplarBasedNN{N <: AbstractNeuralNetwork, FUNI, FUNM, FUNA} <: AbstractNeuralNetwork
    encoder::N
    encoded_exemplars::Matrix{Float32}
    translation::Matrix{Float32}
    interaction_function!::FUNI
    membership_normalization!::FUNM
    activation_function!::FUNA
    states_n::Int

    function ExemplarBasedNN(
        encoder::N,
        encoded_exemplars::Matrix{Float32},  # features x exemplars
        translation::Matrix{Float32};  # actions x exemplars
        interaction_method::Symbol,
        membership_normalization::Symbol,
        activation_function::Symbol,
    ) where N
        states_n = size(encoded_exemplars, 2)
        if interaction_method == :cosine
            encoded_exemplars = normalize_unit(encoded_exemplars)
            interaction_function! = interaction_cosine!
        elseif interaction_method == :mul_norm
            encoded_exemplars = normalize_std(encoded_exemplars)
            interaction_function! = interaction_mul_norm!
        elseif interaction_method == :mul
            interaction_function! = interaction_mul
        else
            throw(ArgumentError("Unknown distance metric: $interaction_method"))
        end

        if membership_normalization == :none
            membership_normalization! = membership_none
        elseif membership_normalization == :unit
            membership_normalization! = membership_unit!
        elseif membership_normalization == :norm
            membership_normalization! = membership_norm!
        elseif membership_normalization == :softmax
            membership_normalization! = membership_softmax!
        elseif occursin("mval", String(membership_normalization))
            m_value = parse(Int, split(String(membership_normalization), "_")[2])
            membership_normalization! = membership_mval_generator(m_value)
        else
            throw(ArgumentError("Unknown membership normalization: $membership_normalization"))
        end

        if activation_function == :none
            activation_function! = activation_none
        elseif activation_function == :softmax
            activation_function! = activation_softmax!
        elseif activation_function == :d_sum
            activation_function! = activation_dsum!
        else
            throw(ArgumentError("Unknown activation function: $activation_function"))
        end

        interaction_fun_type = typeof(interaction_function!)
        membership_fun_type = typeof(membership_normalization!)
        activation_fun_type = typeof(activation_function!)

        return new{N, interaction_fun_type, membership_fun_type, activation_fun_type}(
            encoder,
            encoded_exemplars,
            translation,
            interaction_function!,
            membership_normalization!,
            activation_function!,
            states_n
        )
    end
end

# ------------------------------------------------------------------------------------------
# Interaction functions
function interaction_mul(encoded::Matrix{Float32}, exemplars::Matrix{Float32}) :: Matrix{Float32}
    return exemplars' * encoded
end

function interaction_mul_norm!(encoded::Matrix{Float32}, exemplars::Matrix{Float32}) :: Matrix{Float32}
    normalize_std!(encoded)
    return exemplars' * encoded
end

function interaction_cosine!(encoded::Matrix{Float32}, exemplars::Matrix{Float32}) :: Matrix{Float32}
    normalize_unit!(encoded)
    return exemplars' * encoded
end

# ------------------------------------------------------------------------------------------
# Membership functions
function membership_none(interaction::AbstractVector{Float32})
    return nothing
end

function membership_unit!(interaction::AbstractVector{Float32})
    LinearAlgebra.normalize!(interaction)
end

function membership_norm!(interaction::AbstractVector{Float32})
    mean = Statistics.mean(interaction)
    interaction .-= mean
    std = Statistics.std(interaction)
    interaction .*= 1.0f0 / std
end

function membership_softmax!(interaction::AbstractVector{Float32})
    max_val = typemin(Float32)
    for val in interaction
        @fastmath max_val = ifelse(val > max_val, val, max_val)
    end
    interaction .-= max_val
    broadcast!(exp, interaction, interaction)
    interaction .*= 1.0f0 / sum(interaction)
end

function membership_mval_generator(m_value::Int)
    return (interaction::AbstractVector{Float32}) -> begin
        @fastmath @inbounds @simd for i in eachindex(interaction)
            distance = abs(1.0f0 - interaction[i]) + EPSILON_EXEMPLARBASEDNN
            interaction[i] = (1.0f0 / distance) ^ m_value
        end
        interaction .*= 1.0f0 / sum(interaction)
    end
end

# ------------------------------------------------------------------------------------------
# activation functions
# They do not change anything for simple max choosing
# There will be difference for crossover, I should check how to do it properly
function activation_softmax!(interaction::AbstractVector{Float32})
    max_val = typemin(Float32)
    for val in interaction
        @fastmath max_val = ifelse(val > max_val, val, max_val)
    end
    interaction .-= max_val
    broadcast!(exp, interaction, interaction)
    interaction .*= 1.0f0 / sum(interaction)
end

function activation_none(interaction::AbstractVector{Float32})
    return nothing
end

function activation_dsum!(interaction::AbstractVector{Float32})
    sum_val = sum(interaction)
    interaction .*= 1.0f0 / sum_val
end

# ------------------------------------------------------------------------------------------
# main functions
function predict(nn::ExemplarBasedNN, X) :: Matrix{Float32}
    actions_by_observation = predict_pre_activation(nn, X)
    for col in eachcol(actions_by_observation)
        nn.activation_function!(col)
    end
    return actions_by_observation
end

function predict_pre_activation(nn::ExemplarBasedNN, X) :: Matrix{Float32}
    membership_matrix = membership(nn, X)
    # it will return actions x batch_size
    return nn.translation * membership_matrix
end

function membership(nn::ExemplarBasedNN, X)::Matrix{Float32}
    encoded_x = predict(nn.encoder, X)
    return _encoded_membership!(nn, encoded_x)
end

function encoded_membership(nn::ExemplarBasedNN, encoded_x::Matrix{Float32})::Matrix{Float32}
    encoded_copy = Base.copy(encoded_x)
    return _encoded_membership!(nn, encoded_copy)
end

function _encoded_membership!(nn::ExemplarBasedNN, encoded_x::Matrix{Float32})::Matrix{Float32}
    # interaction matrix - features x exemplars
    interaction_matrix = nn.interaction_function!(encoded_x, nn.encoded_exemplars)
    for col in eachcol(interaction_matrix)
        # we membership normalize it along one exemplar (columns)
        nn.membership_normalization!(col)
    end
    return interaction_matrix
end

function get_states_number(nn::DistanceBasedClassificator)
    return nn.states_n
end

# ------------------------------------------------------------------------------------------
# Other interface functions
function get_neural_network(name::Val{:ExemplarBasedNN})
    return ExemplarBasedNN
end

function get_loss(nn::ExemplarBasedNN) :: Function
    return get_loss(nn.encoder)
end

function get_Flux_representation(nn::ExemplarBasedNN)
    return get_Flux_representation(nn.encoder)
end

# ------------------------------------------------------------------------------------------
# Utils
function normalize_unit(x::Matrix{Float32}) :: Matrix{Float32}
    copied = Base.copy(x)
    normalize_unit!(copied)
    return copied
end

function normalize_unit!(x::Matrix{Float32})
    # for col in eachcol(x)
    #     LinearAlgebra.normalize!(col)
    # end
    for col_ind in axes(x, 2)
        sum_squared = 0.0f0
        @turbo for row_ind in axes(x, 1)
            sum_squared += x[row_ind, col_ind] ^ 2
        end
        sum_squared = 1.0f0 / sqrt(sum_squared)
        @turbo for row_ind in axes(x, 1)
            x[row_ind, col_ind] *= sum_squared
        end
    end
end

function normalize_std(x::Matrix{Float32}) :: Matrix{Float32}
    copied = Base.copy(x)
    normalize_std!(copied)
    return copied
end

function normalize_std!(x::Matrix{Float32})
    for col in eachcol(x)
        mean = Statistics.mean(col)
        col .-= mean
        std = Statistics.std(col)
        col .*= 1.0f0 / std
    end
end
