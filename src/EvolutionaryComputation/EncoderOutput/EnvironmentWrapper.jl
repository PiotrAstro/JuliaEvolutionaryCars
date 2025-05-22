module EnvironmentWrapper

import ..NeuralNetwork
import ..Environment
import ..StatesGrouping

import Statistics
import Dates
import Logging
import Printf

export EnvironmentWrapperStruct, get_action_size, get_fitnesses, copy, is_verbose, set_verbose!, create_new_based_on

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct EnvironmentWrapperStruct
    _envs::Vector{<:Environment.AbstractEnvironment}
    _autoencoder::NeuralNetwork.Autoencoder
    _initial_space_explorers_n::Int
    _max_states_considered::Int
    _run_statistics::Environment.RunStatistics
    _translation_size::Tuple{Int, Int}
    _activation_function::Symbol
    _verbose::Bool
end

# --------------------------------------------------------------------------------------------------
# Public functions

function EnvironmentWrapperStruct(
        envs::Vector{<:Environment.AbstractEnvironment},
        visualization_env::Environment.AbstractEnvironment,
        run_statistics::Environment.RunStatistics=Environment.RunStatistics(),

        ;
        
        encoder_dict::Dict{Symbol,Any},
        decoder_dict::Dict{Symbol,Any},
        autoencoder_dict::Dict{Symbol,<:Any},
        initial_space_explorers_n::Int,
        max_states_considered::Int,
        activation_function::Symbol,
        verbose::Bool=false,
        environment_norm::Union{Nothing, Symbol}=nothing,
        environment_norm_kwargs::Union{Nothing, Dict}=nothing,
    )
    encoder = NeuralNetwork.get_neural_network(encoder_dict[:name])(; encoder_dict[:kwargs]...)
    decoder = NeuralNetwork.get_neural_network(decoder_dict[:name])(; decoder_dict[:kwargs]...)
    autoencoder = NeuralNetwork.Autoencoder(encoder, decoder; autoencoder_dict...)

    # initial state space exploration
    # random NNs creation
    action_n = Environment.get_action_size(envs[1])
    NNs = [NeuralNetwork.Random_NN(action_n, activation_function) for _ in 1:initial_space_explorers_n]
    
    # states collection
    trajectories = _collect_trajectories(envs, NNs, run_statistics)
    states = _combine_states_from_trajectories([(1.0, trajectories)], max_states_considered)

    if !isnothing(environment_norm)
        norm_type = Environment.get_environment(environment_norm)
        if isnothing(environment_norm_kwargs)
            norm_data = Environment.get_norm_data(norm_type, states)
        else
            norm_data = Environment.get_norm_data(norm_type, states; environment_norm_kwargs...)
        end
        states = Environment.norm_ASSEQ(norm_type, norm_data, states)
        envs = [norm_type(env, norm_data) for env in envs]
        visualization_env = norm_type(visualization_env, norm_data)
    end
    
    NeuralNetwork.learn!(autoencoder, states; verbose=verbose)

    if verbose
        Logging.@info "Autoencoder trained"
    end

    encoded_states = NeuralNetwork.predict(encoder, NeuralNetwork.get_sequence_with_ids(states, [1])) # get first state,just to get latent space size
    translation_size = (size(encoded_states, 1), action_n)

    if verbose
        Logging.@info "Exemplars tree created"
    end

    return (
        EnvironmentWrapperStruct(
            envs,
            autoencoder,
            initial_space_explorers_n,
            max_states_considered,
            run_statistics,
            translation_size,
            activation_function,
            verbose
        ),
        visualization_env
    )
end

function get_trajectories(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32})
    return _collect_trajectories(env_wrap._envs, [get_full_NN(env_wrap, translation)], env_wrap._run_statistics)
end

function copy(env_wrap::EnvironmentWrapperStruct)::EnvironmentWrapperStruct
    envs_copy = [Environment.copy(env) for env in env_wrap._envs]
    autoencoder_copy = NeuralNetwork.copy(env_wrap._autoencoder)

    return EnvironmentWrapperStruct(
        envs_copy,
        autoencoder_copy,
        env_wrap._initial_space_explorers_n,
        env_wrap._max_states_considered,
        env_wrap._run_statistics,
        env_wrap._translation_size,
        env_wrap._activation_function,
        env_wrap._verbose
    )
end

function get_action_size(env_wrap::EnvironmentWrapperStruct)::Int
    return env_wrap._translation_size[2]
end

function get_latent_size(env_wrap::EnvironmentWrapperStruct)::Int
    return env_wrap._translation_size[1]
end

function get_fitnesses(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32}) :: Vector{Float64}
    full_NN = get_full_NN(env_wrap, translation)

    envs_copies = [Environment.copy(env) for env in env_wrap._envs]
    result = Environment.get_trajectory_rewards!(envs_copies, full_NN; run_statistics=env_wrap._run_statistics, reset=true)

    return result
end

"""
Create new env_wrapper based on the trajectories and percentages of them.
It will take percent of internal max states considered from each group of trajectories,

args:
env_wrap::EnvironmentWrapperStruct,
trajectories_and_percentages::Vector{Tuple{Float64, Vector{<:Environment.Trajectory}}}

kwargs:
new_n_clusters::Int=-1,

this vector sometimes has a problem with casting on callee side, so we will cast it here
States in trajectories are equal - each states has the same chance of beeing picked (the are added to the same array).

In the future, one can adapt some values of env_wrapper e.g. n_clusters, max_states_considered, fuzzy_logic_of_n_closest etc.
"""
function create_new_based_on(
    env_wrap::EnvironmentWrapperStruct,
    trajectories_and_percentages::Vector{<:Any};
    new_n_clusters::Int=-1,
)
    TSEQ = typeof(trajectories_and_percentages[1][2][1].states) # type of states sequences
    trajectories_and_percentages_casted = Vector{Tuple{Float64,Vector{Environment.Trajectory{TSEQ}}}}(trajectories_and_percentages)

    new_env_wrapper = copy(env_wrap)

    new_states = _combine_states_from_trajectories(trajectories_and_percentages_casted, new_env_wrapper._max_states_considered)
    NeuralNetwork.learn!(new_env_wrapper._autoencoder, new_states; verbose=new_env_wrapper._verbose)

    if env_wrap._verbose
        Logging.@info "Autoencoder retrained"
    end

    return new_env_wrapper
end

# currently hard to translate
# function translate(
#     from_env_wrap::EnvironmentWrapperStruct,
#     from_translation::Matrix{Float32},
#     to_env_wrap::EnvironmentWrapperStruct,
#     to_genes_indices::Vector{Int}=collect(1:to_env_wrap._n_clusters)  # by default translates all genes
# )::Matrix{Float32}  # translated genes, only to_genes_indices are present
#     # create NN

#     from_full_NN = get_full_NN(from_env_wrap, from_translation)
#     to_raw_exemplars = NeuralNetwork.get_sequence_with_ids(to_env_wrap._struct_memory._raw_exemplars, to_genes_indices)
#     return NeuralNetwork.predict_pre_activation(from_full_NN, to_raw_exemplars)
# end

function get_full_NN(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32})
    return NeuralNetwork.EncoderBasedNN(
        env_wrap._autoencoder.encoder,
        translation,
        env_wrap._activation_function
    )
end

function is_verbose(env_wrap::EnvironmentWrapperStruct)::Bool
    return env_wrap._verbose
end

function set_verbose!(env_wrap::EnvironmentWrapperStruct, verbose::Bool)
    env_wrap._verbose = verbose
end

# --------------------------------------------------------------------------------------------------
# Private functions

function _collect_trajectories(envs::Vector{E}, NNs::Vector{<:NeuralNetwork.AbstractAgentNeuralNetwork}, run_statistics::Environment.RunStatistics)::Vector{Environment.Trajectory{SEQ}} where {SEQ<:NeuralNetwork.AbstractStateSequence,E<:Environment.AbstractEnvironment{SEQ}}
    trajectories = Vector{Vector{Environment.Trajectory{SEQ}}}(undef, length(NNs))

    Threads.@threads for i in 1:length(NNs)
        # for i in 1:length(NNs)
        envs_copy = [Environment.copy(env) for env in envs]
        nn = NNs[i]
        trajectories[i] = Environment.get_trajectory_data!(envs_copy, nn; run_statistics=run_statistics, reset=true)
    end
    trajectories_flat = reduce(vcat, trajectories)
    return trajectories_flat
end

function _combine_states_from_trajectories(trajectories_and_percentages::Vector{Tuple{Float64,Vector{Environment.Trajectory{SEQ}}}}, pick_states_n::Int)::SEQ where {SEQ<:NeuralNetwork.AbstractStateSequence}
    @assert sum([percentage for (percentage, _) in trajectories_and_percentages]) ≈ 1.0
    states_to_combine = Vector{SEQ}()

    for (percentage, trajectories) in trajectories_and_percentages
        states_to_pick_n = Int(round(percentage * pick_states_n))
        states_total_n = sum([NeuralNetwork.get_length(trajectory.states) for trajectory in trajectories])
        states_to_pick_n = min(states_to_pick_n, states_total_n)

        states_to_pick = rand(1:states_total_n, states_to_pick_n)
        sequences = Vector{SEQ}()

        from_number = 1
        for i in eachindex(trajectories)
            trajectory = trajectories[i]
            states_one_traj = trajectory.states
            to_number = from_number + NeuralNetwork.get_length(states_one_traj) - 1
            states_to_pick_local = collect(filter(x -> x >= from_number && x <= to_number, states_to_pick))
            states_to_pick_local .-= from_number - 1
            states_one_traj = NeuralNetwork.get_sequence_with_ids(states_one_traj, states_to_pick_local)
            push!(sequences, states_one_traj)
            from_number = to_number + 1
        end

        states_local_combined = SEQ(sequences)
        push!(states_to_combine, states_local_combined)
    end
    states_combined = SEQ(states_to_combine)

    return states_combined
end

end