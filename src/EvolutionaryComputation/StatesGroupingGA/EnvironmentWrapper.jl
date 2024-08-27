module EnvironmentWrapper

import ..NeuralNetwork
import ..Environment
import ..ClusteringHML

import PyCall
println(PyCall.python)
genieclust = PyCall.pyimport("genieclust")
import Statistics
import Clustering
import Distances
using Plots

export EnvironmentWrapperStruct, get_action_size, get_groups_number, get_fitness, actualize!

# --------------------------------------------------------------------------------------------------
# Structs

struct TreeNode
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    distance::Float64
    elements::Vector{Int}
end

mutable struct EnvironmentWrapperStruct
    _envs::Vector{<:Environment.AbstractEnvironment}
    _encoder::NeuralNetwork.AbstractNeuralNetwork
    _autoencoder::NeuralNetwork.AbstractNeuralNetwork
    _game_decoder_data::Tuple{<:Type, Dict{Symbol, Any}}
    _encoded_exemplars::Matrix{Float32}
    _raw_exemplars::Array{Float32}
    _similarity_tree::TreeNode
    _time_tree::TreeNode
    _max_states_considered::Int
end

# --------------------------------------------------------------------------------------------------
# Public functions

function EnvironmentWrapperStruct(
    envs::Vector{<:Environment.AbstractEnvironment},
    encoder_dict::Dict{Symbol, Any},
    decoder_dict::Dict{Symbol, Any},
    game_decoder_dict::Dict{Symbol, Any},
    initial_space_explorers_n::Int,
    max_states_considered::Int
) :: EnvironmentWrapperStruct

    encoder = NeuralNetwork.get_neural_network(encoder_dict[:name])(;encoder_dict[:kwargs]...)
    decoder = NeuralNetwork.get_neural_network(decoder_dict[:name])(;decoder_dict[:kwargs]...)
    autoencoder = NeuralNetwork.Combined_NN([encoder, decoder])

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
    states = _collect_trajectories_states(envs, NNs)

    if size(states, 2) > max_states_considered
        random_columns = rand(1:size(states, 2), max_states_considered)
        states = states[:, random_columns]
    end

    println("States number: $(size(states, 2))")
    # preparing working autoencoder
    NeuralNetwork.learn!(autoencoder, states, states)
    println("Autoencoder trained")
    encoded_states = NeuralNetwork.predict(encoder, states)

    println("Encoded states calculated")

    exemplars_ids, similarity_tree = _get_exemplars(encoded_states)
    println("exemplars number: $(length(exemplars_ids))")
    # scatter!(encoded_states[1, exemplars_ids], encoded_states[2, exemplars_ids], legend=false, color=:red)  # Adds the exemplars to the scatter plot

    # # Saving the plot to a file, e.g., PNG format
    # savefig("scatter_plot.png")
    # println("should be scattered now")

    println("\n\n\n\nfinished\n\n\n\n")
    encoded_exemplars = encoded_states[:, exemplars_ids]
    raw_exemplars = states[:, exemplars_ids]

    return EnvironmentWrapperStruct(
        envs,
        encoder,
        autoencoder,
        (game_decoder_struct, game_decoder_kwargs),
        encoded_exemplars,
        raw_exemplars,
        similarity_tree,
        TreeNode(nothing, nothing, exemplars_ids),
        max_states_considered
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
    full_NN = get_full_NN(env_wrap, genes)
    results = Environment.get_trajectory_rewards!(env_wrap._envs, full_NN)
    return sum(results)
end

"Actualize exemplars and autoencoder based on trajectories generated with provided genes_new_t. Then translates all_solutions to new solutions and returns them."
function actualize!(env_wrap::EnvironmentWrapperStruct, genes_new_trajectories::Vector{Vector{Int}}, all_solutions::Vector{Vector{Int}}) :: Vector{Vector{Int}}
    NNs = Vector{NeuralNetwork.AbstractNeuralNetwork}(undef, length(genes_new_trajectories))

    Threads.@threads for i in 1:length(genes_new_trajectories)
        solution = genes_new_trajectories[i]
        full_NN = get_full_NN(env_wrap, solution)
        NNs[i] = full_NN
    end
    
    states = _collect_trajectories_states(env_wrap._envs, NNs)
    if size(states, 2) > env_wrap._max_states_considered
        random_columns = rand(1:size(states, 2), env_wrap._max_states_considered)
        states = states[:, random_columns]
    end
    println("States number: $(size(states, 2))")

    # We have to retrain autoencoder to cluster new states
    # new_states_old_encoding = NeuralNetwork.predict(env_wrap._encoder, states)
    # NeuralNetwork.learn!(env_wrap._autoencoder, states, states)
    # println("Autoencoder trained")
    # new_encoded_states = NeuralNetwork.predict(env_wrap._encoder, states)

    # # get new exemplars, states and newly encoded states
    # new_exemplars_ids = _get_exemplars(new_encoded_states)
    # println("exemplars number: $(length(new_exemplars_ids))")
    # new_exemplars = new_encoded_states[:, new_exemplars_ids]
    # new_exemplars_old_encoding = new_states_old_encoding[:, new_exemplars_ids]

    # just for testing!!!!!!
    new_states_old_encoding = NeuralNetwork.predict(env_wrap._encoder, states)
    new_exemplars_ids = _get_exemplars(new_states_old_encoding)
    println("exemplars number: $(length(new_exemplars_ids))")
    new_exemplars_old_encoding = new_states_old_encoding[:, new_exemplars_ids]
    new_exemplars_old_encoding = hcat(new_exemplars_old_encoding, env_wrap._encoded_exemplars)
    new_exemplars = new_exemplars_old_encoding
    # end just for testing

    # run through old encoder to get new exemplars encoded in old way, so that I can compare them with old exemplars

    new_solutions = _translate_solutions(env_wrap._encoded_exemplars, new_exemplars_old_encoding, all_solutions)
    env_wrap._encoded_exemplars = new_exemplars

    return new_solutions
end

function get_full_NN(env_wrap::EnvironmentWrapperStruct, genes::Vector{Int}) :: NeuralNetwork.AbstractNeuralNetwork
    # traditional way to do it
    # new_game_decoder = env_wrap._game_decoder_data[1](;env_wrap._game_decoder_data[2]...)

    # # creating one hot encoding
    # actions_n = get_action_size(env_wrap)
    # one_hot = zeros(Float32, actions_n, size(env_wrap._encoded_exemplars, 2))
    # @inbounds for i in 1:size(env_wrap._encoded_exemplars, 2)
    #     one_hot[genes[i], i] = 1.0
    # end

    # NeuralNetwork.learn!(new_game_decoder, env_wrap._encoded_exemplars, one_hot)
    # full_NN = NeuralNetwork.Combined_NN(
    #     [
    #         env_wrap._encoder,
    #         new_game_decoder
    #     ]
    # )

    # return full_NN

    # ---------------------
    # new way to do it
    return NeuralNetwork.DistanceBasedClassificator(
        env_wrap._encoder,
        env_wrap._encoded_exemplars,
        genes,
        get_action_size(env_wrap)
    )
end

# --------------------------------------------------------------------------------------------------
# Private functions

function _collect_trajectories_states(envs::Vector{<:Environment.AbstractEnvironment}, NNs::Vector{<:NeuralNetwork.AbstractNeuralNetwork})
    trajectories_states = Vector{Array{Float32}}(undef, length(NNs))

    Threads.@threads for i in 1:length(NNs)
        nn = NNs[i]
        trajectories = Environment.get_trajectory_data!(envs, nn)
        trajectories_states[i] = reduce(hcat, [trajectory.states for trajectory in trajectories])
    end

    states = reduce(hcat, trajectories_states)
    return states
end

function _translate_solutions(old_exemplars::Matrix{Float32}, new_exemplars::Matrix{Float32}, all_solutions::Vector{Vector{Int}}) :: Vector{Vector{Int}}
    # calculate simmilarity matrix
    distances_matrix = [vec(sum((old_exemplars .- new_exemplar).^2, dims=1)) for new_exemplar in eachcol(new_exemplars)]
    display(distances_matrix)
    new_to_old = argmin.(distances_matrix)
    display(new_to_old)

    # translate solutions
    new_solutions = Vector{Vector{Int}}(undef, length(all_solutions))
    Threads.@threads for i in 1:length(all_solutions)
        solution = all_solutions[i]
        new_solution = [solution[old] for old in new_to_old]
        new_solutions[i] = new_solution
    end

    return new_solutions
end

include("_EnvironmentWrapperClustering.jl")

end