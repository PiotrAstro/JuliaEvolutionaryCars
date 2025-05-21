
module Environment

using TestItems

import Pkg
import MuJoCo
MuJoCo.init_visualiser()
import Random
import LinearAlgebra
import SimpleDirectMediaLayer as SDL
import SimpleDirectMediaLayer.LibSDL2 as LSDL2
import Statistics

import ..NeuralNetwork

DATA_DIR = joinpath(dirname(@__FILE__), "..", "..", "data")

export AbstractEnvironment, copy, reset!, react!, get_state, get_action_size, is_alive, get_trajectory_data!, get_trajectory_rewards!, get_environment, prepare_environments_kwargs, visualize!, get_ASSEQ, get_norm_data, norm_ASSEQ, AbstractNormEnvironmentWrapper, RunStatistics, get_statistics
# ------------------------------------------------------------------------------------------------

abstract type AbstractEnvironment{ASSEQ <: NeuralNetwork.AbstractStateSequence} end

function get_ASSEQ(env::AbstractEnvironment{ASSEQ}) where {ASSEQ<:NeuralNetwork.AbstractStateSequence}
    return ASSEQ
end

struct Trajectory{ASSEQ<:NeuralNetwork.AbstractStateSequence}
    states::ASSEQ
    actions::Matrix{Float32}
    rewards::Vector{Float64}
    rewards_sum::Float64
end

function Trajectory(states::ASSEQ, actions::Matrix{Float32}, rewards::Vector{Float64}) where {ASSEQ<:NeuralNetwork.AbstractStateSequence}
    rewards_sum = sum(rewards)
    return Trajectory{ASSEQ}(states, actions, rewards, rewards_sum)
end

"""
This one is used as holder only for states, other parts are not valid
"""
function Trajectory(states::ASSEQ) where {ASSEQ<:NeuralNetwork.AbstractStateSequence}
    return Trajectory{ASSEQ}(states, Matrix{Float32}(undef, 0, 0), Vector{Float64}(), 0.0)
end

# ------------------------------------------------------------------------------------------------
# Interface functions

function get_environment(environment::Val{T}) where T
    throw("unimplemented")
end

"Doesnt reset environment afterwards, real implementation will have some kwargs"
function visualize!(env::AbstractEnvironment, model::NeuralNetwork.AbstractAgentNeuralNetwork, parent_env=env, reset::Bool = true;)
    throw("unimplemented")
end

function get_action_size(env::AbstractEnvironment)::Int
    throw("unimplemented")
end

# function get_safe_data(env::AbstractEnvironment)::Dict{Symbol}
#     throw("unimplemented")
# end

# function load_safe_data!(env::AbstractEnvironment, data::Dict{Symbol}) 
#     throw("unimplemented")
# end

function reset!(env::AbstractEnvironment)
    throw("unimplemented")
end

function react!(env::AbstractEnvironment, actions::AbstractVector{Float32}) :: Float64
    throw("unimplemented")
end

function get_state(env::AbstractEnvironment{ASSEQ}) :: INTERNAL where {INTERNAL, ASSEQ<:NeuralNetwork.AbstractStateSequence{INTERNAL}}
    throw("unimplemented")
end

function is_alive(env::AbstractEnvironment)::Bool
    throw("unimplemented")
end

function copy(env::AbstractEnvironment)
    throw("unimplemented")
end


# ------------------------------------------------------------------------------------------------
# Normalization wrapper for environments
abstract type AbstractNormEnvironmentWrapper{ASSEQ} <: AbstractEnvironment{ASSEQ} end

# calculates normalization parameters and should be used in constructor of the wraper itself
function get_norm_data(::Type{ANEW}, data::ASSEQ; kwargs...) where {ASSEQ, ANEW<:AbstractNormEnvironmentWrapper{ASSEQ}}
    throw("unimplemented")
end

function (::Type{ANEW})(env::AbstractEnvironment, norm_data) where {ANEW<:AbstractNormEnvironmentWrapper}
    throw("unimplemented")
end

function norm_ASSEQ(::Type{ANEW}, norm_data, asseq::ASSEQ)::ASSEQ where {ASSEQ<:NeuralNetwork.AbstractStateSequence, ANEW<:AbstractNormEnvironmentWrapper{ASSEQ}}
    throw("unimplemented")
end


"""
A thread safe structure to store run statistics.
- Frames are single observations of the environment (those received by Agents)
- Evaluations are number of times get_trajectory_rewards! or get_trajectory_data! was called
- Trajectories are number of trajectories that were generated or could be (so e.g. for get_trajectory_rewards! with 5 environments it will be 5)

Total vs Collected
- Total are from get_trajectory_rewards! + get_trajectory_data! calls
- Collected are from get_trajectory_data! only

Important!
- structure is implementation detail, you should just pass it to get_trajectory_rewards! and get_trajectory_data! functions
- get results using get_statistics function
"""
@kwdef struct RunStatistics
    total_frames::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    total_evaluations::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    total_trajectories::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    collected_frames::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    collected_evaluations::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    collected_trajectories::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
end

function get_statistics(run_statistics::RunStatistics)
    return (;
        total_frames = run_statistics.total_frames[],
        total_evaluations = run_statistics.total_evaluations[],
        total_trajectories = run_statistics.total_trajectories[],
        collected_frames = run_statistics.collected_frames[],
        collected_evaluations = run_statistics.collected_evaluations[],
        collected_trajectories = run_statistics.collected_trajectories[]
    )
end

# using BenchmarkTools
# import Profile
# import PProf
# b_states(envs_alive, ASSEQ) = ASSEQ([get_state(env) for (env, _) in envs_alive])
# function b_states_just_calc(envs_alive)
#     for (env, _) in envs_alive
#         get_state(env)
#     end
# end


# Some general functions, not interface functions
"""
Get the rewards of the trajectory of the environments using the neural network. Returns sum of rewards for each environment. Modifies state of environments - resets them before and leaves them used
"""
function get_trajectory_rewards!(
        envs::Vector{E},
        neural_network::Union{NeuralNetwork.AbstractAgentNeuralNetwork, NeuralNetwork.AbstractTrainableAgentNeuralNetwork};
        run_statistics::Union{Nothing, RunStatistics} = nothing,
        reset::Bool = true
    ) :: Vector{Float64} where {INTERNAL, ASSEQ<:NeuralNetwork.AbstractStateSequence{INTERNAL}, E<:AbstractEnvironment{ASSEQ}}
    rewards = zeros(Float64, length(envs))

    if reset
        for env in envs
            reset!(env)
        end
    end

    envs_alive = [(env, i) for (i, env) in enumerate(envs) if is_alive(env)]
    trajectories_n = length(envs_alive)

    # benchmarks:
    # states = ASSEQ([get_state(env) for (env, _) in envs_alive])
    # actions = NeuralNetwork.predict(neural_network, states)
    # println("ASSEQ creation:")
    # b = @benchmark b_states($envs_alive, $ASSEQ)
    # display(b)

    # println("ASSEQ just calc:")
    # b = @benchmark b_states_just_calc($envs_alive)
    # display(b)

    # env_tmp = envs_alive[1][1]
    # actions_view = view(actions, :, 1)
    # b = @benchmark react!($env_tmp, $actions_view)
    # println("react! creation:")
    # display(b)

    # println("NeuralNetwork predict:")
    # b = @benchmark NeuralNetwork.predict($neural_network, $states)
    # display(b)
    # # Profile.clear()
    # # Profile.@profile for i in 1:100000
    # #     states = ASSEQ([get_state(env) for (env, _) in envs_alive])
    # #     actions = NeuralNetwork.predict(neural_network, get_nn_input(states))
    # # end
    # # PProf.pprof(;webport=2137)
    # throw("fdsfdsvsdf")

    frames_n = 0

    while length(envs_alive) > 0
        states = ASSEQ([get_state(env) for (env, _) in envs_alive])
        actions = NeuralNetwork.predict(neural_network, states)
        i = 1
        while i <= length(envs_alive)
            (env, j) = envs_alive[i]
            rewards[j] += react!(env, view(actions, :, i))
            frames_n += 1
            if !is_alive(env)
                deleteat!(envs_alive, i)
                actions = actions[:, 1:end .!= i]
            else
                i += 1
            end
        end
    end

    if run_statistics !== nothing
        Threads.atomic_add!(run_statistics.total_frames, frames_n)
        Threads.atomic_add!(run_statistics.total_evaluations, 1)
        Threads.atomic_add!(run_statistics.total_trajectories, trajectories_n)
    end

    return rewards
end

"""
Get the rewards, states and actions of the trajectory of the environments using the neural network. Returns sum of rewards for each environment. Modifies state of environments - resets them before and leaves them used
"""
function get_trajectory_data!(
        envs::Vector{E},
        neural_network::Union{NeuralNetwork.AbstractAgentNeuralNetwork, NeuralNetwork.AbstractTrainableAgentNeuralNetwork};
        run_statistics::Union{Nothing, RunStatistics} = nothing,
        reset::Bool = true
    ) :: Vector{Trajectory{ASSEQ}} where {INTERNAL, ASSEQ<:NeuralNetwork.AbstractStateSequence{INTERNAL}, E<:AbstractEnvironment{ASSEQ}}
    trajectory_data = Vector{Tuple{Vector{Float64}, Vector{INTERNAL}, Vector{Vector{Float32}}}}()
    for env in envs
        if reset
            reset!(env)
        end

        push!(trajectory_data, (Vector{Float64}(), Vector{INTERNAL}(), Vector{Vector{Float32}}()))
    end
    envs_alive = [(env, i) for (i, env) in enumerate(envs) if is_alive(env)]
    trajectory_n = length(envs_alive)
    frames_n = 0

    while length(envs_alive) > 0
        states = ASSEQ([get_state(env) for (env, _) in envs_alive])
        actions = NeuralNetwork.predict(neural_network, states)
        i = 1
        while i <= length(envs_alive)
            (env, j) = envs_alive[i]
            current_action = actions[:, i]
            reward = react!(env, current_action)

            frames_n += 1

            push!(trajectory_data[j][1], reward)
            push!(trajectory_data[j][2], NeuralNetwork.copy_nth_state(states, i))
            push!(trajectory_data[j][3], current_action)

            if !is_alive(env)
                deleteat!(envs_alive, i)
                states = NeuralNetwork.remove_nth_state(states, i)
                actions = actions[:, 1:end .!= i]
            else
                i += 1
            end
        end
    end

    if run_statistics !== nothing
        Threads.atomic_add!(run_statistics.total_frames, frames_n)
        Threads.atomic_add!(run_statistics.total_evaluations, 1)
        Threads.atomic_add!(run_statistics.total_trajectories, trajectory_n)
        Threads.atomic_add!(run_statistics.collected_frames, frames_n)
        Threads.atomic_add!(run_statistics.collected_evaluations, 1)
        Threads.atomic_add!(run_statistics.collected_trajectories, trajectory_n)
    end

    return [Trajectory(ASSEQ(states), reduce(hcat, actions), rewards) for (rewards, states, actions) in trajectory_data]
end


# includes
include("_CarEnvironment/_CarEnvironment.jl")
include("_Humanoid/_Humanoid.jl")
include("_HalfCheetah/_HalfCheetah.jl")

# normalization wrapper for environments
include("_NormMatrixASSEQ/_NormMatrixASSEQ.jl")


# functions using includes
function get_environment(name::Symbol)
    return get_environment(Val(name))
end

function prepare_environments_kwargs(dict_universal::Dict, dict_changeable::Vector{D}) where D<:Dict
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
