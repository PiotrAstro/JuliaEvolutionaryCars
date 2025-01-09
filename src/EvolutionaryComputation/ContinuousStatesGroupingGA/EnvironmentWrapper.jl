module EnvironmentWrapper

import ..NeuralNetwork
import ..Environment
import ..StatesGrouping

import Statistics
import Clustering
import Distances
import Plots
import Dates
import JLD
import Logging
using StatsPlots

export EnvironmentWrapperStruct, get_action_size, get_groups_number, get_fitness, copy, is_verbose, set_verbose!, translate, create_new_based_on, create_time_distance_tree, get_genes

# --------------------------------------------------------------------------------------------------
# Structs


mutable struct EnvironmentWrapperStruct
    _envs::Vector{<:Environment.AbstractEnvironment}
    _n_clusters::Int
    _encoder::NeuralNetwork.AbstractNeuralNetwork
    _decoder::NeuralNetwork.AbstractNeuralNetwork
    _autoencoder::NeuralNetwork.AbstractNeuralNetwork
    _game_decoder_struct
    _game_decoder_kwargs::Dict
    _encoded_exemplars::Matrix{Float32}
    _raw_exemplars::Environment.AbstractStateSequence
    _similarity_tree::StatesGrouping.TreeNode
    _max_states_considered::Int
    _distance_metric::Symbol  # :euclidean or :cosine or :cityblock
    _exemplars_clustering::Symbol  # :genieclust or :pam or :kmedoids
    _hclust_distance::Symbol  # :ward or :single or :complete or :average
    _hclust_time::Symbol  # :ward or :single or :complete or :average
    _verbose::Bool
end

# --------------------------------------------------------------------------------------------------
# Public functions

function EnvironmentWrapperStruct(
    envs::Vector{<:Environment.AbstractEnvironment};

    encoder_dict::Dict{Symbol, Any},
    decoder_dict::Dict{Symbol, Any},
    autoencoder_dict::Dict{Symbol, <:Any},
    game_decoder_dict::Dict{Symbol, Any},
    initial_space_explorers_n::Int,
    max_states_considered::Int,
    n_clusters::Int,
    distance_metric::Symbol = :cosine,
    exemplars_clustering::Symbol = :genieclust,
    hclust_distance::Symbol = :ward,
    hclust_time::Symbol = :ward,
    verbose::Bool = false
) :: EnvironmentWrapperStruct

    encoder = NeuralNetwork.get_neural_network(encoder_dict[:name])(;encoder_dict[:kwargs]...)
    decoder = NeuralNetwork.get_neural_network(decoder_dict[:name])(;decoder_dict[:kwargs]...)
    autoencoder = NeuralNetwork.Autoencoder(encoder, decoder; autoencoder_dict...)

    game_decoder_struct = NeuralNetwork.get_neural_network(game_decoder_dict[:name])
    game_decoder_kwargs = game_decoder_dict[:kwargs]

    # initial state space exploration
    # random NNs creation
    NNs = [
        get_full_NN(encoder, new_game_decoder(game_decoder_struct, game_decoder_kwargs)) for _ in 1:initial_space_explorers_n
    ]
    # states collection
    trajectories = _collect_trajectories(envs, NNs)
    states = _combine_states_from_trajectories([(1.0, trajectories)], max_states_considered)
    states_nn_input = Environment.get_nn_input(states)
    NeuralNetwork.learn!(autoencoder, states_nn_input, states_nn_input; verbose=verbose)

    if verbose
        Logging.@info "Autoencoder trained"
    end

    encoded_states = NeuralNetwork.predict(encoder, states_nn_input)
    exemplars_ids, similarity_tree = StatesGrouping.get_exemplars(encoded_states, n_clusters; distance_metric, exemplars_clustering, hclust_distance)
    encoded_exemplars = encoded_states[:, exemplars_ids]
    states_exeplars = Environment.get_sequence_with_ids(states, exemplars_ids)

    if verbose
        Logging.@info "Exemplars tree created"
    end

    return EnvironmentWrapperStruct(
        envs,
        n_clusters,
        encoder,
        decoder,
        autoencoder,
        game_decoder_struct,
        game_decoder_kwargs,
        encoded_exemplars,
        states_exeplars,
        similarity_tree,
        max_states_considered,
        distance_metric,
        exemplars_clustering,
        hclust_distance,
        hclust_time,
        verbose
    )
end

"""
returns Tuple{Vector{Trajectory}, TreeNode}
"""
function create_time_distance_tree(env_wrap::EnvironmentWrapperStruct, game_decoder::NeuralNetwork.AbstractNeuralNetwork)
    trajectories = _collect_trajectories(env_wrap._envs, [get_full_NN(env_wrap, game_decoder)])
    states_in_trajectories = [trajectory.states for trajectory in trajectories]
    encoded_states_by_trajectory = [NeuralNetwork.predict(env_wrap._encoder, Environment.get_nn_input(states_one_traj)) for states_one_traj in states_in_trajectories]
    return trajectories, StatesGrouping.create_time_distance_tree(encoded_states_by_trajectory, env_wrap._encoded_exemplars, env_wrap._hclust_time)
end

function copy(env_wrap::EnvironmentWrapperStruct) :: EnvironmentWrapperStruct
    envs_copy = [Environment.copy(env) for env in env_wrap._envs]
    autoencoder_copy = NeuralNetwork.copy(env_wrap._autoencoder)
    encoder_copy = autoencoder_copy.encoder
    decoder_copy = autoencoder_copy.decoder

    return EnvironmentWrapperStruct(
        envs_copy,
        env_wrap._n_clusters,
        encoder_copy,
        decoder_copy,
        autoencoder_copy,
        env_wrap._game_decoder_struct,
        env_wrap._game_decoder_kwargs,
        env_wrap._encoded_exemplars,
        env_wrap._raw_exemplars,
        env_wrap._similarity_tree,
        env_wrap._max_states_considered,
        env_wrap._distance_metric,
        env_wrap._exemplars_clustering,
        env_wrap._hclust_distance,
        env_wrap._hclust_time,
        env_wrap._verbose
    )
end

function get_action_size(env_wrap::EnvironmentWrapperStruct) :: Int
    return Environment.get_action_size(env_wrap._envs[1])
end

function get_groups_number(env_wrap::EnvironmentWrapperStruct) :: Int
    return size(env_wrap._encoded_exemplars, 2)
end

"""
Learn game_decoder on env_wrap exemplars and goal_responses, evaluate and return fitness and game_decoder.
returns Tuple{Float64, <:NeuralNetwork.AbstractNeuralNetwork}
"""
function get_fitness(env_wrap::EnvironmentWrapperStruct, game_decoder::NeuralNetwork.AbstractNeuralNetwork, goal_responses::Matrix{Float32})
    game_decoder_copy = NeuralNetwork.copy(game_decoder)
    NeuralNetwork.learn!(game_decoder_copy, env_wrap._encoded_exemplars, goal_responses; verbose=false)
    return get_fitness(env_wrap, game_decoder_copy), game_decoder_copy
end

function get_fitness(env_wrap::EnvironmentWrapperStruct, game_decoder::NeuralNetwork.AbstractNeuralNetwork) :: Float64
    full_NN = get_full_NN(env_wrap, game_decoder)

    envs_copies = [Environment.copy(env) for env in env_wrap._envs]
    result = sum(Environment.get_trajectory_rewards!(envs_copies, full_NN))

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
    ) :: EnvironmentWrapperStruct
    if new_n_clusters == -1
        new_n_clusters = env_wrap._n_clusters
    end
    TSEQ = typeof(trajectories_and_percentages[1][2][1].states) # type of states sequences
    trajectories_and_percentages_casted = Vector{Tuple{Float64, Vector{Environment.Trajectory{TSEQ}}}}(trajectories_and_percentages)

    new_env_wrapper = copy(env_wrap)

    new_states = _combine_states_from_trajectories(trajectories_and_percentages_casted, new_env_wrapper._max_states_considered)
    new_states_nn_input = Environment.get_nn_input(new_states)
    NeuralNetwork.learn!(new_env_wrapper._autoencoder, new_states_nn_input, new_states_nn_input; verbose=new_env_wrapper._verbose)

    if env_wrap._verbose
        Logging.@info "Autoencoder retrained"
    end

    new_encoded_states = NeuralNetwork.predict(new_env_wrapper._encoder, new_states_nn_input)

    # # get new exemplars, states and newly encoded states
    new_exemplars_ids, new_similarity_tree = StatesGrouping.get_exemplars(new_encoded_states, new_n_clusters; distance_metric=env_wrap._distance_metric, exemplars_clustering=env_wrap._exemplars_clustering, hclust_distance=env_wrap._hclust_distance)
    new_exemplars = new_encoded_states[:, new_exemplars_ids]
    new_raw_exemplars = Environment.get_sequence_with_ids(new_states, new_exemplars_ids)

    new_env_wrapper._encoded_exemplars = new_exemplars
    new_env_wrapper._raw_exemplars = new_raw_exemplars
    new_env_wrapper._similarity_tree = new_similarity_tree
    new_env_wrapper._n_clusters = new_n_clusters

    return new_env_wrapper
end

function translate(
        from_env_wrap::EnvironmentWrapperStruct,
        from_game_decoder::NeuralNetwork.AbstractNeuralNetwork,
        to_env_wrap::EnvironmentWrapperStruct,
        to_genes_indices::Vector{Int} = collect(1:to_env_wrap._n_clusters)  # by default translates all genes
    ) :: Matrix{Float32}  # translated genes, only to_genes_indices are present
    # create NN

    from_full_NN = get_full_NN(from_env_wrap, from_game_decoder)
    to_raw_exemplars = Environment.get_sequence_with_ids(to_env_wrap._raw_exemplars, to_genes_indices)
    return NeuralNetwork.predict(from_full_NN, Environment.get_nn_input(to_raw_exemplars))
end

function get_genes(env_wrap::EnvironmentWrapperStruct, game_decoder::NeuralNetwork.AbstractNeuralNetwork) :: Matrix{Float32}
    return NeuralNetwork.predict(game_decoder, env_wrap._encoded_exemplars)
end

function get_full_NN(env_wrap::EnvironmentWrapperStruct, game_decoder::NeuralNetwork.AbstractNeuralNetwork)
    return get_full_NN(env_wrap._encoder, game_decoder)
end

function get_full_NN(encoder::NeuralNetwork.AbstractNeuralNetwork, game_decoder::NeuralNetwork.AbstractNeuralNetwork)
    return NeuralNetwork.Combined_NN([
        encoder,
        game_decoder
    ])
end

function new_game_decoder(env_wrap::EnvironmentWrapperStruct) :: NeuralNetwork.AbstractNeuralNetwork
    return new_game_decoder(env_wrap._game_decoder_struct, env_wrap._game_decoder_kwargs)
end

function new_game_decoder(game_decoder_struct, game_decoder_kwargs) :: NeuralNetwork.AbstractNeuralNetwork
    return game_decoder_struct(;game_decoder_kwargs...)
end

function is_verbose(env_wrap::EnvironmentWrapperStruct) :: Bool
    return env_wrap._verbose
end

function set_verbose!(env_wrap::EnvironmentWrapperStruct, verbose::Bool)
    env_wrap._verbose = verbose
end

# --------------------------------------------------------------------------------------------------
# Private functions

function _collect_trajectories(envs::Vector{E}, NNs::Vector{<:NeuralNetwork.AbstractNeuralNetwork}) :: Vector{Environment.Trajectory{SEQ}} where {SEQ<:Environment.AbstractStateSequence, E<:Environment.AbstractEnvironment{SEQ}}
    trajectories = Vector{Vector{Environment.Trajectory{SEQ}}}(undef, length(NNs))

    Threads.@threads for i in 1:length(NNs)
    # for i in 1:length(NNs)
        envs_copy = [Environment.copy(env) for env in envs]
        nn = NNs[i]
        trajectories[i] = Environment.get_trajectory_data!(envs_copy, nn)
    end
    trajectories_flat = reduce(vcat, trajectories)
    return trajectories_flat
end

function _combine_states_from_trajectories(trajectories_and_percentages::Vector{Tuple{Float64, Vector{Environment.Trajectory{SEQ}}}}, pick_states_n::Int) :: SEQ where {SEQ<:Environment.AbstractStateSequence}
    @assert sum([percentage for (percentage, _) in trajectories_and_percentages]) ≈ 1.0
    states_to_combine = Vector{SEQ}()

    for (percentage, trajectories) in trajectories_and_percentages
        states_to_pick_n = Int(round(percentage * pick_states_n))
        states_total_n = sum([Environment.get_length(trajectory.states) for trajectory in trajectories])
        states_to_pick_n = min(states_to_pick_n, states_total_n)

        states_local_combined = SEQ([trajectory.states for trajectory in trajectories])
        states_to_pick = rand(1:states_total_n, states_to_pick_n)
        states_local_combined = Environment.get_sequence_with_ids(states_local_combined, states_to_pick)
        push!(states_to_combine, states_local_combined)
    end
    states_combined = SEQ(states_to_combine)

    return states_combined
end

end