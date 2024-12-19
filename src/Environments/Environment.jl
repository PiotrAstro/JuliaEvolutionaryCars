
module Environment

import ..NeuralNetwork

export AbstractEnvironment, get_safe_data, load_safe_data!, copy, reset!, react!, get_state, get_state_size, get_action_size, is_alive, get_trajectory_data!, get_trajectory_rewards!, get_environment, prepare_environments_kwargs, visualize!

# concrete implementation should have fist parametric type to be a number of dimensions
abstract type AbstractEnvironment{IDN} end

struct Trajectory{IDN, ODN}
    states::Array{Float32, IDN}
    actions::Array{Float32, ODN}
    rewards::Vector{Float64}
    rewards_sum::Float64
end

function Trajectory(states::Array{Float32}, actions::Array{Float32}, rewards::Array{Float64})
    rewards_sum = sum(rewards)
    return Trajectory(states, actions, rewards, rewards_sum)
end

# ------------------------------------------------------------------------------------------------
# Interface functions

"Doesnt reset environment afterwards, real implementation will have some kwargs"
function visualize!(env::AbstractEnvironment, model::NeuralNetwork.AbstractNeuralNetwork, reset::Bool = true;)
    throw("unimplemented")
end

function get_state_dimmensions_number(env::AbstractEnvironment)::Int
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

function react!(env::AbstractEnvironment, actions::AbstractVector{Float32}) :: Float64
    throw("unimplemented")
end

function get_state(env::AbstractEnvironment) :: Vector{Float32}
    throw("unimplemented")
end

function is_alive(env::AbstractEnvironment)::Bool
    throw("unimplemented")
end

function copy(env::AbstractEnvironment)
    throw("unimplemented")
end

# ------------------------------------------------------------------------------------------------

# Some general functions, not interface functions

"Get the rewards of the trajectory of the environments using the neural network. Returns sum of rewards for each environment. Modifies state of environments - resets them before and leaves them used"
function get_trajectory_rewards!(envs::Vector{E}, neural_network::NN; reset::Bool = true) :: Vector{Float64} where {E<:AbstractEnvironment, ODN, NN<:NeuralNetwork.AbstractNeuralNetwork{ODN}}
    rewards = zeros(Float64, length(envs))

    if reset
        for env in envs
            reset!(env)
        end
    end

    envs_alive = [(env, i) for (i, env) in enumerate(envs) if is_alive(env)]

    while length(envs_alive) > 0
        states = reduce(hcat, [get_state(env) for (env, _) in envs_alive])
        actions = NeuralNetwork.predict(neural_network, states)
        i = 1
        while i <= length(envs_alive)
            (env, j) = envs_alive[i]
            rewards[j] += react!(env, copy_slice_at_last_position(actions, i))
            if !is_alive(env)
                deleteat!(envs_alive, i)
                actions = remove_slice_at_last_position(actions, i)
                i -= 1
            end
            i += 1
        end
    end

    return rewards
end

"Get the rewards, states and actions of the trajectory of the environments using the neural network. Returns sum of rewards for each environment. Modifies state of environments - resets them before and leaves them used"
function get_trajectory_data!(
        envs::Vector{E},
        neural_network::NN,
        reset::Bool = true
    ) :: Vector{Trajectory{IDN+1, ODN}} where {IDN, ODN, E<:AbstractEnvironment{IDN}, NN<:NeuralNetwork.AbstractNeuralNetwork{ODN}}
    state_dimensions_n = get_state_size(envs[1])
    nn_output_dimensions_n = NeuralNetwork.get_output_dimensions_number(neural_network)
    trajectory_data = Vector{Tuple{Vector{Float64}, Vector{Array{Float32, IDN}}, Vector{Array{Float32, ODN}}}}()
    for env in envs
        if reset
            reset!(env)
        end

        push!(trajectory_data, (Vector{Float64}(), Vector{Array{Float32, IDN}}(), Vector{Array{Float32, ODN}}()))
    end
    envs_alive = [(env, i) for (i, env) in enumerate(envs) if is_alive(env)]

    while length(envs_alive) > 0
        states = reduce(hcat, [get_state(env) for (env, _) in envs_alive])
        actions = NeuralNetwork.predict(neural_network, states)
        i = 1
        while i <= length(envs_alive)
            (env, j) = envs_alive[i]
            current_action = copy_slice_at_last_position(actions, i)
            reward = react!(env, current_action)
            push!(trajectory_data[j][1], reward)
            push!(trajectory_data[j][2], copy_slice_at_last_position(states, i))
            push!(trajectory_data[j][3], current_action)

            if !is_alive(env)
                deleteat!(envs_alive, i)
                states = remove_slice_at_last_position(states, i)
                actions = remove_slice_at_last_position(actions, i)
                i -= 1
            end
            i += 1
        end
    end
    return [Trajectory(cat(states..., dims=(IDN+1)), cat(actions..., dims=(ODN+1)), rewards) for (rewards, states, actions) in trajectory_data]
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
# utils for states and actions

@generated function copy_slice_at_last_position(arr::AbstractArray{T,N}, i::Int) where {T,N}
    # Construct the indexing expression at compile time
    exprs = [:(:) for _ in 1:(N-1)]  # for all dims except the last, just `:`
    # For the last dimension, we use i
    return quote
        arr[$(exprs...), i]
    end
end

function outer_length(arr::AbstractArray{T,N}) :: Int where {T,N}
    return size(arr, N)
end

@generated function copy_slice_at_positions(arr::AbstractArray{T,N}, positions::Vector{Int}) where {T,N}
    # Construct the indexing expression at compile time
    exprs = [:(:) for _ in 1:(N-1)]  # for all dims except the last, just `:`
    # For the last dimension, we use i
    return quote
        arr[$(exprs...), positions]
    end
end

@generated function remove_slice_at_last_position(arr::AbstractArray{T,N}, i::Int) where {T,N}
    # Construct the indexing expression at compile time
    exprs = [:(:) for _ in 1:(N-1)]  # for all dims except the last, just `:`
    # For the last dimension, we use `remaining`
    return quote
        remaining = [1:i-1; i+1:size(arr, N)]
        arr[$(exprs...), remaining]
    end
end



end # module
