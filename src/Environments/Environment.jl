
module Environment

import ..NeuralNetwork

export AbstractEnvironment, get_safe_data, load_safe_data!, copy, reset!, react!, get_state, get_state_size, get_action_size, is_alive, get_trajectory_data!, get_trajectory_rewards!, get_environment, prepare_environments_kwargs, visualize!
export AbstractStateSequence, get_length, copy_nth_state, get_sequence_with_ids, remove_nth_state, get_nn_input

# ------------------------------------------------------------------------------------------------
# state sequences managers

# INTERNAL - internal concrete type, environment will receive it for reaction, e.g. Vector{Float32} or Array{Float32, 3} for rgb images
abstract type AbstractStateSequence{INTR} end

function (::AbstractStateSequence{INTERNAL})(states::Vector{INTERNAL}) :: AbstractStateSequence{INTERNAL} where {INTERNAL}
    throw("unimplemented")
end

function (::AbstractStateSequence{INTERNAL})(seqs::Vector{AbstractStateSequence{INTERNAL}}) :: AbstractStateSequence{INTERNAL} where {INTERNAL}
    throw("unimplemented")
end

function get_length(seq::AbstractStateSequence{INTERNAL}) :: Int where {INTERNAL}
    throw("unimplemented")
end

function copy_nth_state(seq::AbstractStateSequence{INTERNAL}, n::Int) :: INTERNAL where {INTERNAL} 
    throw("unimplemented")
end

function get_sequence_with_ids(seq::AbstractStateSequence{INTERNAL}, ids::AbstractVector{Int}) :: AbstractStateSequence{INTERNAL} where {INTERNAL}
    throw("unimplemented")
end

function remove_nth_state(seq::AbstractStateSequence{INTERNAL}, n::Int) :: AbstractStateSequence{INTERNAL} where {INTERNAL}
    throw("unimplemented")
end

function get_nn_input(seq::AbstractStateSequence{INTERNAL}) where {INTERNAL}
    throw("unimplemented")
end

# ------------------------------------------------------------------------------------------------

abstract type AbstractEnvironment{ASS <: AbstractStateSequence} end

struct Trajectory{ASS<:AbstractStateSequence}
    states::ASS
    actions::Matrix{Float32}
    rewards::Vector{Float64}
    rewards_sum::Float64
end

function Trajectory(states::ASS, actions::Matrix{Float32}, rewards::Vector{Float64}) where {ASS<:AbstractStateSequence}
    rewards_sum = sum(rewards)
    return Trajectory{ASS}(states, actions, rewards, rewards_sum)
end

# ------------------------------------------------------------------------------------------------
# Interface functions

"Doesnt reset environment afterwards, real implementation will have some kwargs"
function visualize!(env::AbstractEnvironment, model::NeuralNetwork.AbstractNeuralNetwork, reset::Bool = true;)
    throw("unimplemented")
end

function get_state_size(env::AbstractEnvironment)::Vector{Int}
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

function react!(env::AbstractEnvironment{AbstractStateSequence{INTERNAL}}, actions::INTERNAL) :: Float64 where {INTERNAL}
    throw("unimplemented")
end

function get_state(env::AbstractEnvironment{AbstractStateSequence{INTERNAL}}) :: INTERNAL where {INTERNAL}
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
function get_trajectory_rewards!(envs::Vector{E}, neural_network::NN; reset::Bool = true) :: Vector{Float64} where {INTERNAL, ASS<:AbstractStateSequence{INTERNAL}, E<:AbstractEnvironment{ASS}, NN<:NeuralNetwork.AbstractNeuralNetwork}
    rewards = zeros(Float64, length(envs))

    if reset
        for env in envs
            reset!(env)
        end
    end

    envs_alive = [(env, i) for (i, env) in enumerate(envs) if is_alive(env)]

    while length(envs_alive) > 0
        states = ASS([get_state(env) for (env, _) in envs_alive])
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
        neural_network::NN,
        reset::Bool = true
    ) :: Vector{Trajectory{ASS}} where {INTERNAL, ASS<:AbstractStateSequence{INTERNAL}, E<:AbstractEnvironment{ASS}, NN<:NeuralNetwork.AbstractNeuralNetwork}
    trajectory_data = Vector{Tuple{Vector{Float64}, Vector{INTERNAL}, Vector{Vector{Float32}}}}()
    for env in envs
        if reset
            reset!(env)
        end

        push!(trajectory_data, (Vector{Float64}(), Vector{INTERNAL}(), Vector{Vector{Float32}}()))
    end
    envs_alive = [(env, i) for (i, env) in enumerate(envs) if is_alive(env)]

    while length(envs_alive) > 0
        states = ASS([get_state(env) for (env, _) in envs_alive])
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
    return [Trajectory(ASS(states), hcat(actions...), rewards) for (rewards, states, actions) in trajectory_data]
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
