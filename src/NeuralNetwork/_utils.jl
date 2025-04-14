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
        return Lux.sigmoid, true  # element-wise, can be in layer
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
