
module Environment

import ..NeuralNetwork

export AbstractEnvironment, get_safe_data, load_safe_data!, copy, reset!, react!, get_state, get_state_size, get_action_size, is_alive, get_trajectory_data!, get_trajectory_rewards!, get_environment, prepare_environments_kwargs, visualize!
export AbstractStateSequence, get_length, copy_nth_state, get_sequence_with_ids, remove_nth_state, get_nn_input

# ------------------------------------------------------------------------------------------------
# state sequences managers

# INTERNAL - internal concrete type, environment will receive it for reaction, e.g. Vector{Float32} or Array{Float32, 3} for rgb images
abstract type AbstractStateSequence{INTR} end

# AbstractStateSequence should have the same type in return
function (::AbstractStateSequence{INTERNAL})(states::Vector{INTERNAL}) :: AbstractStateSequence{INTERNAL} where {INTERNAL}
    throw("unimplemented")
end

# AbstractStateSequence should have the same type in return
function (::AbstractStateSequence{INTERNAL})(seqs::Vector{AbstractStateSequence{INTERNAL}}) :: AbstractStateSequence{INTERNAL} where {INTERNAL}
    throw("unimplemented")
end

function get_length(seq::ASSEQ) :: Int where {ASSEQ<:AbstractStateSequence}
    throw("unimplemented")
end

function copy_nth_state(seq::ASSEQ, n::Int) :: INTERNAL where {INTERNAL, ASSEQ<:AbstractStateSequence{INTERNAL}} 
    throw("unimplemented")
end

function get_sequence_with_ids(seq::ASSEQ, ids::AbstractVector{Int}) :: ASSEQ where {ASSEQ<:AbstractStateSequence}
    throw("unimplemented")
end

# It shouldnt change the original sequence
function remove_nth_state(seq::ASSEQ, n::Int) :: ASSEQ where {ASSEQ<:AbstractStateSequence}
    throw("unimplemented")
end

function get_nn_input(seq::ASSEQ) where {ASSEQ<:AbstractStateSequence}
    throw("unimplemented")
end

# ------------------------------------------------------------------------------------------------

abstract type AbstractEnvironment{ASSEQ <: AbstractStateSequence} end

struct Trajectory{ASSEQ<:AbstractStateSequence}
    states::ASSEQ
    actions::Matrix{Float32}
    rewards::Vector{Float64}
    rewards_sum::Float64
end

function Trajectory(states::ASSEQ, actions::Matrix{Float32}, rewards::Vector{Float64}) where {ASSEQ<:AbstractStateSequence}
    rewards_sum = sum(rewards)
    return Trajectory{ASSEQ}(states, actions, rewards, rewards_sum)
end

# ------------------------------------------------------------------------------------------------
# Interface functions

"Doesnt reset environment afterwards, real implementation will have some kwargs"
function visualize!(env::AbstractEnvironment, model::NeuralNetwork.AbstractNeuralNetwork, reset::Bool = true;)
    throw("unimplemented")
end

function get_action_size(env::AbstractEnvironment)::Int
    throw("unimplemented")
end

function get_safe_data(env::AbstractEnvironment)::Dict{Symbol}
    throw("unimplemented")
end

function load_safe_data!(env::AbstractEnvironment, data::Dict{Symbol}) 
    throw("unimplemented")
end

function reset!(env::AbstractEnvironment)
    throw("unimplemented")
end

function react!(env::AbstractEnvironment, actions::AbstractVector{Float32}) :: Float64
    throw("unimplemented")
end

function get_state(env::AbstractEnvironment{ASSEQ}) :: INTERNAL where {INTERNAL, ASSEQ<:AbstractStateSequence{INTERNAL}}
    throw("unimplemented")
end

function is_alive(env::AbstractEnvironment)::Bool
    throw("unimplemented")
end

function copy(env::AbstractEnvironment)
    throw("unimplemented")
end

# Some general functions, not interface functions

"Get the rewards of the trajectory of the environments using the neural network. Returns sum of rewards for each environment. Modifies state of environments - resets them before and leaves them used"
function get_trajectory_rewards!(envs::Vector{E}, neural_network::NeuralNetwork.AbstractNeuralNetwork; reset::Bool = true) :: Vector{Float64} where {INTERNAL, ASSEQ<:AbstractStateSequence{INTERNAL}, E<:AbstractEnvironment{ASSEQ}}
    rewards = zeros(Float64, length(envs))

    if reset
        for env in envs
            reset!(env)
        end
    end

    envs_alive = [(env, i) for (i, env) in enumerate(envs) if is_alive(env)]

    while length(envs_alive) > 0
        states = ASSEQ([get_state(env) for (env, _) in envs_alive])
        actions = NeuralNetwork.predict(neural_network, get_nn_input(states))
        i = 1
        while i <= length(envs_alive)
            (env, j) = envs_alive[i]
            rewards[j] += react!(env, @view actions[:, i])
            if !is_alive(env)
                deleteat!(envs_alive, i)
                actions = actions[:, 1:end .!= i]
                i -= 1
            end
            i += 1
        end
    end

    return rewards
end

"""
Get the rewards, states and actions of the trajectory of the environments using the neural network. Returns sum of rewards for each environment. Modifies state of environments - resets them before and leaves them used
"""
function get_trajectory_data!(
        envs::Vector{E},
        neural_network::NeuralNetwork.AbstractNeuralNetwork,
        reset::Bool = true
    ) :: Vector{Trajectory{ASSEQ}} where {INTERNAL, ASSEQ<:AbstractStateSequence{INTERNAL}, E<:AbstractEnvironment{ASSEQ}}
    trajectory_data = Vector{Tuple{Vector{Float64}, Vector{INTERNAL}, Vector{Vector{Float32}}}}()
    for env in envs
        if reset
            reset!(env)
        end

        push!(trajectory_data, (Vector{Float64}(), Vector{INTERNAL}(), Vector{Vector{Float32}}()))
    end
    envs_alive = [(env, i) for (i, env) in enumerate(envs) if is_alive(env)]

    while length(envs_alive) > 0
        states = ASSEQ([get_state(env) for (env, _) in envs_alive])
        actions = NeuralNetwork.predict(neural_network, get_nn_input(states))
        i = 1
        while i <= length(envs_alive)
            (env, j) = envs_alive[i]
            current_action = actions[:, i]
            reward = react!(env, current_action)
            push!(trajectory_data[j][1], reward)
            push!(trajectory_data[j][2], copy_nth_state(states, i))
            push!(trajectory_data[j][3], current_action)

            if !is_alive(env)
                deleteat!(envs_alive, i)
                states = remove_nth_state(states, i)
                actions = actions[:, 1:end .!= i]
                i -= 1
            end
            i += 1
        end
    end
    return [Trajectory(ASSEQ(states), hcat(actions...), rewards) for (rewards, states, actions) in trajectory_data]
end


# includes
include("_CarEnvironment/_CarEnvironment.jl")



# functions using includes
function get_environment(name::Symbol) :: Type
    if name == :BasicCarEnvironment
        return BasicCarEnvironment
    else
        throw("Environment not found")
    end
end

function prepare_environments_kwargs(dict_universal::Dict{Symbol, Any}, dict_changeable::Vector{Dict{Symbol, Any}}) :: Vector{Dict{Symbol, Any}}
    dicts_copy = [deepcopy(dict_universal) for _ in 1:length(dict_changeable)]
    for i in 1:length(dict_changeable)
        for (key, value) in dict_changeable[i]
            dicts_copy[i][key] = value
        end
    end

    return dicts_copy
end

# ------------------------------------------------------------------------------------------------


end # module
