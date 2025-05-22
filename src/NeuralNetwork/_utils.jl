# --------------------------------------------------------------------------------
# Some math utils

global const EPSILON_F32::Float32 = Float32(1e-7)

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
        @fastmath sum_squared = 1.0f0 / sqrt(sum_squared)
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
    @fastmath for col in eachcol(x)
        mean = Statistics.mean(col)
        col .-= mean
        std = Statistics.std(col)
        col .*= 1.0f0 / std
    end
end

function softmax!(x::AbstractVector{Float32})
    max_val = typemin(Float32)
    for val in x
        @fastmath max_val = ifelse(val > max_val, val, max_val)
    end
    @fastmath x .-= max_val
    @fastmath broadcast!(exp, x, x)
    @fastmath x .*= 1.0f0 / sum(x)
end


# --------------------------------------------------------------------------------
# functions to change from SimpleChains to Lux and vice versa


function _set_params!(internal_params::Vector, params::AbstractArray, start_pos::Int=1)::Int
    params_vec = vec(params)
    end_pos = start_pos + length(params_vec) - 1
    internal_params[start_pos:end_pos] .= params_vec
    return end_pos + 1
end

function _set_params!(internal_params::Vector, params::NamedTuple, start_pos::Int=1)::Int
    for param in values(params)
        start_pos = _set_params!(internal_params, param, start_pos)
    end
    return start_pos
end

function _copy_params!(params::AbstractArray, internal_params::Vector, start_pos::Int=1)::Int
    params_vec = vec(params)
    end_pos = start_pos + length(params_vec) - 1
    params_vec .= view(internal_params, start_pos:end_pos)
    return end_pos + 1
end

function _copy_params!(params::NamedTuple, internal_params::Vector, start_pos::Int=1)::Int
    for param in values(params)
        start_pos = _copy_params!(param, internal_params, start_pos)
    end
    return start_pos
end


# ---------------------------------------------------------------------------------------
# Activation and Loss functions

"""
Get activation function from symbol
    
Returns function and information if it should be applied element-wise(True), and to whole array(False).
"""
function _get_activation_function(name::Symbol)::Tuple{Function, Bool}
    if name == :relu
        return Lux.relu, true  # element-wise, can be in layer
    elseif name == :sigmoid
        return Lux.sigmoid_fast, true  # element-wise, can be in layer
    elseif name == :tanh
        return Lux.tanh_fast, true  # element-wise, can be in layer
    elseif name == :none
        return identity, true  # element-wise, can be in layer
    elseif name == :softmax
        return Lux.softmax, false  # applies to whole array, must be outside layer
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
        return Lux.CrossEntropyLoss(; logits=Val(false))  # it means that we should use softmax before
    elseif name == :mse
        return Lux.MSELoss()
    elseif name == :mae
        return Lux.MAELoss()
    elseif name == :kldivergence
        return Lux.KLDivergenceLoss()
    else
        throw("Loss function not implemented")
    end
end



# ------------------------------------------------------------------------------------------
# activation functions
# They do not change anything for simple max choosing
# There will be difference for crossover, I should check how to do it properly
function get_activation_layer_function(activation_function::Symbol)
    if activation_function == :none
        activation_function! = activation_none
    elseif activation_function == :softmax
        activation_function! = activation_softmax!
    elseif activation_function == :d_sum
        activation_function! = activation_dsum!
    elseif activation_function == :tanh
        activation_function! = activation_tanh!
    elseif activation_function == :sigmoid
        activation_function! = activation_sigmoid!
    else
        throw(ArgumentError("Unknown activation function: $activation_function"))
    end
    return activation_function!
end

function activation_softmax!(interaction::AbstractVector{Float32})
    softmax!(interaction)
end

function activation_tanh!(interaction::AbstractVector{Float32})
    broadcast!(Lux.tanh_fast, interaction, interaction)
end

function activation_sigmoid!(interaction::AbstractVector{Float32})
    broadcast!(Lux.sigmoid_fast, interaction, interaction)
end

function activation_none(interaction::AbstractVector{Float32})
    return nothing
end

function activation_dsum!(interaction::AbstractVector{Float32})
    sum_val = sum(interaction)
    interaction .*= 1.0f0 / sum_val
end
