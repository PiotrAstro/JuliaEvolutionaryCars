struct NormMatrixASSEQ{E<:AbstractEnvironment{NeuralNetwork.MatrixASSEQ}} <: AbstractNormEnvironmentWrapper{NeuralNetwork.MatrixASSEQ}
    env::E
    means::Vector{Float32}
    stds_inverse::Vector{Float32}
end

function get_norm_data(::Type{NormMatrixASSEQ}, data::NeuralNetwork.MatrixASSEQ; epsilon::Float32=0.1f0)
    states = data.states
    means = [Float32(Statistics.mean(row)) for row in eachrow(states)]
    stds = [Float32(Statistics.std(row)) for row in eachrow(states)]
    for i in eachindex(stds)
        stds[i] = 1.0f0 / (stds[i] + epsilon)
    end

    return (;means=means, stds_inverse=stds)
end

function NormMatrixASSEQ(env::AbstractEnvironment{NeuralNetwork.MatrixASSEQ}, norm_data)
    means = norm_data.means
    stds_inverse = norm_data.stds_inverse
    return NormMatrixASSEQ(env, means, stds_inverse)
end

function norm_ASSEQ(::Type{NormMatrixASSEQ}, norm_data, matrix_asseq::NeuralNetwork.MatrixASSEQ)::NeuralNetwork.MatrixASSEQ
    states_copy = Base.copy(matrix_asseq.states)

    for col in eachcol(states_copy)
        col .-= norm_data.means
        col .*= norm_data.stds_inverse
    end

    return NeuralNetwork.MatrixASSEQ(states_copy)
end

# -----------------------------------------------------------------------------
# AbstractEnvironment functions

function get_environment(environment::Val{:NormMatrixASSEQ})
    return NormMatrixASSEQ
end

"Doesnt reset environment afterwards, real implementation will have some kwargs"
function visualize!(env::NormMatrixASSEQ, model::NeuralNetwork.AbstractAgentNeuralNetwork, parent_env=env, reset::Bool = true;kwargs...)
    visualize!(env.env, model, parent_env, reset; kwargs...)
end

function get_action_size(env::NormMatrixASSEQ)::Int
    return get_action_size(env.env)
end

# function get_safe_data(env::NormMatrixASSEQ)::Dict{Symbol}
#     throw("unimplemented")
# end

# function load_safe_data!(env::NormMatrixASSEQ, data::Dict{Symbol}) 
#     throw("unimplemented")
# end

function reset!(env::NormMatrixASSEQ)
    return reset!(env.env)
end

function react!(env::NormMatrixASSEQ, actions::AbstractVector{Float32}) :: Float64
    return react!(env.env, actions)
end

function get_state(env::NormMatrixASSEQ) :: Vector{Float32}
    state = get_state(env.env)
    state .-= env.means
    state .*= env.stds_inverse

    return state
end

function is_alive(env::NormMatrixASSEQ)::Bool
    return is_alive(env.env)
end

function copy(env::NormMatrixASSEQ)
    return NormMatrixASSEQ(copy(env.env), env.means, env.stds_inverse)
end