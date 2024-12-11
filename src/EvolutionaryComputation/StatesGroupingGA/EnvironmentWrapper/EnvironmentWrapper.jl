module EnvironmentWrapper

import ..NeuralNetwork
import ..Environment

import PyCall
import Statistics
import Clustering
import Distances
import Plots
import Dates
import JLD
import Logging

Logging.@info PyCall.python * "\n"
genieclust = PyCall.pyimport("genieclust")

export EnvironmentWrapperStruct, get_action_size, get_groups_number, get_fitness, actualize!, copy, translate_solutions, is_verbose, set_verbose!

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct TreeNode
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    distance::Float64
    elements::Vector{Int}
end

mutable struct EnvironmentWrapperStruct
    _envs::Vector{<:Environment.AbstractEnvironment}
    _n_clusters::Int
    _encoder::NeuralNetwork.AbstractNeuralNetwork
    _decoder::NeuralNetwork.AbstractNeuralNetwork
    _autoencoder::NeuralNetwork.AbstractNeuralNetwork
    _game_decoder_data::Tuple{<:Type, Dict{Symbol, Any}}
    _encoded_exemplars::Matrix{Float32}
    _encoded_exemplars_normalised::Matrix{Float32}
    _raw_all_states::Array{Float32}
    _raw_exemplars::Matrix{Float32}
    _similarity_tree::TreeNode
    _time_distance_tree::TreeNode
    _max_states_considered::Int
    _fuzzy_logic_of_n_closest::Int
    _result_memory::Dict{Vector{Int}, Float64}
    _result_memory_mutex::ReentrantLock
    _verbose::Bool
end

# --------------------------------------------------------------------------------------------------
# Public functions

function EnvironmentWrapperStruct(
    envs::Vector{<:Environment.AbstractEnvironment},
    encoder_dict::Dict{Symbol, Any},
    decoder_dict::Dict{Symbol, Any},
    autoencoder_dict::Dict{Symbol, <:Any},
    game_decoder_dict::Dict{Symbol, Any},
    initial_space_explorers_n::Int,
    max_states_considered::Int,
    n_clusters::Int,
    fuzzy_logic_of_n_closest::Int,
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
        NeuralNetwork.Combined_NN([
            encoder,
            (game_decoder_struct)(;game_decoder_kwargs...)
        ]) for _ in 1:initial_space_explorers_n
    ]
    # states collection
    states, states_by_trajectories = _collect_trajectories_states(envs, NNs)

    if size(states, 2) > max_states_considered
        random_columns = rand(1:size(states, 2), max_states_considered)
        states = states[:, random_columns]
    end

    NeuralNetwork.learn!(autoencoder, states, states; verbose=verbose)
    if verbose
        Logging.@info "Autoencoder trained"
    end
    encoded_states_by_trajectory = [NeuralNetwork.predict(encoder, states_one_traj) for states_one_traj in states_by_trajectories]
    encoded_states = NeuralNetwork.predict(encoder, states)

    exemplars_ids, similarity_tree = _get_exemplars(encoded_states, encoder, n_clusters)

    encoded_exemplars = encoded_states[:, exemplars_ids]
    time_distance_tree = _create_time_distance_tree(encoded_states_by_trajectory, encoded_exemplars)

    if verbose
        Logging.@info "Exemplars and time distance tree created"
    end
    

    wrapper = EnvironmentWrapperStruct(
        envs,
        n_clusters,
        encoder,
        decoder,
        autoencoder,
        (game_decoder_struct, game_decoder_kwargs),
        encoded_exemplars,
        _normalize_exemplars(encoded_exemplars),
        states,
        states[:, exemplars_ids],
        similarity_tree,
        time_distance_tree,
        max_states_considered,
        fuzzy_logic_of_n_closest,
        Dict{Vector{Int}, Float64}(),
        ReentrantLock(),
        verbose
    )

    return wrapper
end

function copy(env_wrap::EnvironmentWrapperStruct) :: EnvironmentWrapperStruct
    envs_copy = [Environment.copy(env) for env in env_wrap._envs]
    autoencoder_copy = NeuralNetwork.copy(env_wrap._autoencoder)
    encoder_copy = autoencoder_copy.encoder
    decoder_copy = autoencoder_copy.decoder

    lock(env_wrap._result_memory_mutex)
    result_memory_copy = Base.copy(env_wrap._result_memory)
    unlock(env_wrap._result_memory_mutex)

    return EnvironmentWrapperStruct(
        envs_copy,
        env_wrap._n_clusters,
        encoder_copy,
        decoder_copy,
        autoencoder_copy,
        env_wrap._game_decoder_data,
        env_wrap._encoded_exemplars,
        env_wrap._encoded_exemplars_normalised,
        env_wrap._raw_all_states,
        env_wrap._raw_exemplars,
        env_wrap._similarity_tree,
        env_wrap._time_distance_tree,
        env_wrap._max_states_considered,
        env_wrap._fuzzy_logic_of_n_closest,
        result_memory_copy,
        ReentrantLock(),
        env_wrap._verbose
    )
end

function is_leaf(tree::TreeNode) :: Bool
    return isnothing(tree.left) && isnothing(tree.right)
end

function get_action_size(env_wrap::EnvironmentWrapperStruct) :: Int
    return Environment.get_action_size(env_wrap._envs[1])
end

function get_groups_number(env_wrap::EnvironmentWrapperStruct) :: Int
    return size(env_wrap._encoded_exemplars, 2)
end


function get_fitness(env_wrap::EnvironmentWrapperStruct, genes::Vector{Int}) :: Float64
    copied_genes = Base.copy(genes)

    lock(env_wrap._result_memory_mutex)
    result = get(env_wrap._result_memory, copied_genes, nothing)
    unlock(env_wrap._result_memory_mutex)

    if isnothing(result)
        full_NN = get_full_NN(env_wrap, copied_genes)
        envs_copies = [Environment.copy(env) for env in env_wrap._envs]
        result = sum(Environment.get_trajectory_rewards!(envs_copies, full_NN))

        lock(env_wrap._result_memory_mutex)
        env_wrap._result_memory[copied_genes] = result
        unlock(env_wrap._result_memory_mutex)
    end

    return result
end

"Actualize exemplars and autoencoder based on trajectories generated with provided genes_new_t. Then translates all_solutions to new solutions and returns them."
function actualize!(env_wrap::EnvironmentWrapperStruct, genes_new_trajectories::Vector{Vector{Int}}, all_solutions::Vector{Vector{Int}}) :: Vector{Vector{Int}}
    NNs = Vector{NeuralNetwork.AbstractNeuralNetwork}(undef, length(genes_new_trajectories))

    # Threads.@threads for i in 1:length(genes_new_trajectories)
    for i in 1:length(genes_new_trajectories)
        solution = genes_new_trajectories[i]
        full_NN = get_full_NN(env_wrap, solution)
        NNs[i] = full_NN
    end
    
    states, states_by_trajectories = _collect_trajectories_states(env_wrap._envs, NNs)
    states = hcat(states, env_wrap._raw_all_states)
    if size(states, 2) > env_wrap._max_states_considered
        random_columns = rand(1:size(states, 2), env_wrap._max_states_considered)
        states = states[:, random_columns]
    end

    # We have to retrain autoencoder to cluster new states
    new_states_old_encoding = NeuralNetwork.predict(env_wrap._encoder, states)
    NeuralNetwork.learn!(env_wrap._autoencoder, states, states)

    if env_wrap._verbose
        Logging.@info "Autoencoder retrained"
    end

    new_encoded_states_by_trajectory = [NeuralNetwork.predict(env_wrap._encoder, states_one_traj) for states_one_traj in states_by_trajectories]
    new_encoded_states = NeuralNetwork.predict(env_wrap._encoder, states)

    # # get new exemplars, states and newly encoded states
    new_exemplars_ids, similarity_tree = _get_exemplars(new_encoded_states, env_wrap._encoder, env_wrap._n_clusters)
    new_exemplars = new_encoded_states[:, new_exemplars_ids]
    time_distance_tree = _create_time_distance_tree(new_encoded_states_by_trajectory, new_exemplars)
    new_exemplars_old_encoding = new_states_old_encoding[:, new_exemplars_ids]

    new_solutions = _translate_solutions(env_wrap._encoded_exemplars, new_exemplars_old_encoding, all_solutions)
    env_wrap._encoded_exemplars = new_exemplars
    env_wrap._encoded_exemplars_normalised = _normalize_exemplars(new_exemplars)
    env_wrap._raw_all_states = states
    env_wrap._similarity_tree = similarity_tree
    env_wrap._time_distance_tree = time_distance_tree
    env_wrap._result_memory = Dict{Vector{Int}, Float64}()
    env_wrap._result_memory_mutex = ReentrantLock()

    return new_solutions
end

function get_full_NN(env_wrap::EnvironmentWrapperStruct, genes::Vector{Int}) :: NeuralNetwork.AbstractNeuralNetwork
    return NeuralNetwork.DistanceBasedClassificator(
        env_wrap._encoder,
        env_wrap._encoded_exemplars_normalised,  # env_wrap._encoded_exemplars, # env_wrap._encoded_exemplars_normalised,
        genes,
        get_action_size(env_wrap),
        env_wrap._fuzzy_logic_of_n_closest
    )
end

function translate_solutions(env_wrap_old::EnvironmentWrapperStruct, env_wrap_new::EnvironmentWrapperStruct, all_solutions::Vector{Vector{Int}}) :: Vector{Vector{Int}}
    return _translate_solutions(env_wrap_old._encoded_exemplars, NeuralNetwork.predict(env_wrap_old._encoder, env_wrap_old._raw_exemplars), all_solutions)
end

function is_verbose(env_wrap::EnvironmentWrapperStruct) :: Bool
    return env_wrap._verbose
end

function set_verbose!(env_wrap::EnvironmentWrapperStruct, verbose::Bool)
    env_wrap._verbose = verbose
end

# --------------------------------------------------------------------------------------------------
# Private functions

function _collect_trajectories_states(envs::Vector{<:Environment.AbstractEnvironment}, NNs::Vector{<:NeuralNetwork.AbstractNeuralNetwork}) :: Tuple{Array{Float32}, Vector{Array{Float32}}}
    trajectories_states_separately = Vector{Vector{Array{Float32}}}(undef, length(NNs))
    trajectories_states = Vector{Array{Float32}}(undef, length(NNs))

    # Threads.@threads for i in 1:length(NNs)
    for i in 1:length(NNs)
        envs_copy = [Environment.copy(env) for env in envs]
        nn = NNs[i]
        trajectories = Environment.get_trajectory_data!(envs_copy, nn)
        trajectories_states_separately[i] = [trajectory.states for trajectory in trajectories]
        trajectories_states[i] = reduce(hcat, trajectories_states_separately[i])
    end
    states_by_trajectory_flat = reduce(vcat, trajectories_states_separately)
    states = reduce(hcat, trajectories_states)
    return states, states_by_trajectory_flat
end

function _normalize_exemplars(exemplars::Matrix{Float32}) :: Matrix{Float32}
    sums = sum(exemplars .^ 2, dims=1)
    exemplars_normalised = exemplars ./ sqrt.(sums)
    exemplars_normalised_transposed = exemplars_normalised'
    return exemplars_normalised_transposed
end

function _translate_solutions(old_exemplars::Matrix{Float32}, new_exemplars::Matrix{Float32}, all_solutions::Vector{Vector{Int}}) :: Vector{Vector{Int}}
    # calculate simmilarity matrix
    distances_matrix = Distances.pairwise(Distances.CosineDist(), old_exemplars, new_exemplars)
    new_to_old = [argmin(one_col) for one_col in eachcol(distances_matrix)]

    # translate solutions
    new_solutions = Vector{Vector{Int}}(undef, length(all_solutions))
    # Threads.@threads for i in 1:length(all_solutions)
    for i in 1:length(all_solutions)
        solution = all_solutions[i]
        new_solution = [solution[old] for old in new_to_old]
        new_solutions[i] = new_solution
    end

    return new_solutions
end

include("_EnvironmentWrapperClustering.jl")
include("_EnvironmentWrapperTimeClustering.jl")

end