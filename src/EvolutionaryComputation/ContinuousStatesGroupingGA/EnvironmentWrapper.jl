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
import LinearAlgebra
using StatsPlots

export EnvironmentWrapperStruct, get_action_size, get_groups_number, get_fitness, copy, is_verbose, set_verbose!, translate, create_new_based_on, create_time_distance_tree, normalize_genes_min_0!

# --------------------------------------------------------------------------------------------------
# Structs


mutable struct EnvironmentWrapperStruct
    _envs::Vector{<:Environment.AbstractEnvironment}
    _n_clusters::Int
    _encoder::NeuralNetwork.AbstractNeuralNetwork
    _decoder::NeuralNetwork.AbstractNeuralNetwork
    _autoencoder::NeuralNetwork.AbstractNeuralNetwork
    _encoded_exemplars::Matrix{Float32}
    _raw_exemplars::Environment.AbstractStateSequence
    _similarity_tree::StatesGrouping.TreeNode
    _max_states_considered::Int
    _distance_metric::Symbol  # :euclidean or :cosine or :cityblock
    _exemplars_clustering::Symbol  # :genieclust or :pam or :kmedoids
    _hclust_distance::Symbol  # :ward or :single or :complete or :average
    _hclust_time::Symbol  # :ward or :single or :complete or :average
    _m_value::Int
    _create_time_distance_tree
    _verbose::Bool
end



# --------------------------------------------------------------------------------------------------
# Public functions

function EnvironmentWrapperStruct(
        envs::Vector{<:Environment.AbstractEnvironment}; encoder_dict::Dict{Symbol,Any},
        decoder_dict::Dict{Symbol,Any},
        autoencoder_dict::Dict{Symbol,<:Any},
        initial_space_explorers_n::Int,
        max_states_considered::Int,
        n_clusters::Int,
        distance_metric::Symbol=:cosine,
        exemplars_clustering::Symbol=:genieclust,
        hclust_distance::Symbol=:ward,
        hclust_time::Symbol=:ward,
        time_distance_tree::Symbol=:mine,  # :mine or :markov
        m_value::Int=1,
        verbose::Bool=false
    ) :: EnvironmentWrapperStruct
    if time_distance_tree == :mine
        create_time_distance_tree = StatesGrouping.create_time_distance_tree_mine
    elseif time_distance_tree == :markov
        create_time_distance_tree = StatesGrouping.create_time_distance_tree_markov_fundamental
    else
        throw(ArgumentError("time_distance_tree must be :mine or :markov"))
    end

    encoder = NeuralNetwork.get_neural_network(encoder_dict[:name])(; encoder_dict[:kwargs]...)
    decoder = NeuralNetwork.get_neural_network(decoder_dict[:name])(; decoder_dict[:kwargs]...)
    autoencoder = NeuralNetwork.Autoencoder(encoder, decoder; autoencoder_dict...)

    # initial state space exploration
    # random NNs creation
    action_n = Environment.get_action_size(envs[1])
    NNs = [
        NeuralNetwork.Random_NN(action_n) for _ in 1:initial_space_explorers_n
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
        encoded_exemplars,
        states_exeplars,
        similarity_tree,
        max_states_considered,
        distance_metric,
        exemplars_clustering,
        hclust_distance,
        hclust_time,
        m_value,
        create_time_distance_tree,
        verbose
    )
end

"""
returns Tuple{Vector{Trajectory}, TreeNode}
"""
function create_time_distance_tree(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32})
    trajectories = _collect_trajectories(env_wrap._envs, [get_full_NN(env_wrap, translation)])
    states_in_trajectories = [trajectory.states for trajectory in trajectories]
    full_nn = get_full_NN(env_wrap, translation)
    memberships_by_trajectory = [NeuralNetwork.membership(full_nn, Environment.get_nn_input(states_one_traj)) for states_one_traj in states_in_trajectories]
    return trajectories, env_wrap._create_time_distance_tree(memberships_by_trajectory, env_wrap._hclust_time)
end

function get_trajectories(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32})
    return _collect_trajectories(env_wrap._envs, [get_full_NN(env_wrap, translation)])
end

function copy(env_wrap::EnvironmentWrapperStruct)::EnvironmentWrapperStruct
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
        env_wrap._encoded_exemplars,
        env_wrap._raw_exemplars,
        env_wrap._similarity_tree,
        env_wrap._max_states_considered,
        env_wrap._distance_metric,
        env_wrap._exemplars_clustering,
        env_wrap._hclust_distance,
        env_wrap._hclust_time,
        env_wrap._m_value,
        create_time_distance_tree,
        env_wrap._verbose
    )
end


"""
It normalizes genes by making it non negative and sum to 0
opposed to normalize_genes_min_0! if input is e.g. 0.2 0.4 0.4 it will stay the same
"""
function normalize_genes!(genes::Matrix{Float32})
    for col in eachcol(genes)
        min_value = minimum(col)
        if min_value < 0
            col .+= abs(min_value)
        end
        col ./= sum(col)
    end
end

"""
It normalizes genes by subtracting smallest value from each col and then subtracting by sum
if input is e.g. 0.2 0.4 0.4 it will be normalized to 0 0.5 0.5
"""
function normalize_genes_min_0!(genes::Matrix{Float32})
    for col in eachcol(genes)
        normalize_genes_min_0!(col)
    end
end

function normalize_genes_min_0!(gene::AbstractVector{Float32})
    min_value = minimum(gene)
    gene .-= min_value
    gene ./= sum(gene)
end

function get_action_size(env_wrap::EnvironmentWrapperStruct)::Int
    return Environment.get_action_size(env_wrap._envs[1])
end

function get_groups_number(env_wrap::EnvironmentWrapperStruct)::Int
    return size(env_wrap._encoded_exemplars, 2)
end

function get_fitness(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32}) :: Float64
    full_NN = get_full_NN(env_wrap, translation)

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
)::EnvironmentWrapperStruct
    if new_n_clusters == -1
        new_n_clusters = env_wrap._n_clusters
    end
    TSEQ = typeof(trajectories_and_percentages[1][2][1].states) # type of states sequences
    trajectories_and_percentages_casted = Vector{Tuple{Float64,Vector{Environment.Trajectory{TSEQ}}}}(trajectories_and_percentages)

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
    from_translation::Matrix{Float32},
    to_env_wrap::EnvironmentWrapperStruct,
    to_genes_indices::Vector{Int}=collect(1:to_env_wrap._n_clusters)  # by default translates all genes
)::Matrix{Float32}  # translated genes, only to_genes_indices are present
    # create NN

    from_full_NN = get_full_NN(from_env_wrap, from_translation)
    to_raw_exemplars = Environment.get_sequence_with_ids(to_env_wrap._raw_exemplars, to_genes_indices)
    return NeuralNetwork.predict(from_full_NN, Environment.get_nn_input(to_raw_exemplars))
end

function get_full_NN(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32})
    return NeuralNetwork.DistanceBasedClassificator(
        env_wrap._encoder,
        env_wrap._encoded_exemplars,
        translation,
        get_action_size(env_wrap),
        -1,
        env_wrap._distance_metric,
        env_wrap._m_value
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

function _collect_trajectories(envs::Vector{E}, NNs::Vector{<:NeuralNetwork.AbstractNeuralNetwork})::Vector{Environment.Trajectory{SEQ}} where {SEQ<:Environment.AbstractStateSequence,E<:Environment.AbstractEnvironment{SEQ}}
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

function _combine_states_from_trajectories(trajectories_and_percentages::Vector{Tuple{Float64,Vector{Environment.Trajectory{SEQ}}}}, pick_states_n::Int)::SEQ where {SEQ<:Environment.AbstractStateSequence}
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