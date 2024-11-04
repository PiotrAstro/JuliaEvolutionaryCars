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
import Plots
import Dates
import JLD

export EnvironmentWrapperStruct, get_action_size, get_groups_number, get_fitness, actualize!, copy, translate_solutions

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
    _encoder::NeuralNetwork.AbstractNeuralNetwork
    _decoder::NeuralNetwork.AbstractNeuralNetwork
    _autoencoder::NeuralNetwork.AbstractNeuralNetwork
    _game_decoder_data::Tuple{<:Type, Dict{Symbol, Any}}
    _encoded_exemplars::Matrix{Float32}
    _encoded_exemplars_normalised::Matrix{Float32}
    _raw_all_states::Array{Float32}
    _raw_exemplars::Matrix{Float32}
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

    

    # load states from log/states_ 1 to 9 and encode them
    # good_states_splitted = [JLD.load("log/states_$i.jld", "states") for i in 1:9]
    # println("splitted states")
    # for state in good_states_splitted
    #     display(state)
    # end
    # action_taken = zeros(Int, sum([size(states, 2) for states in good_states_splitted]))
    # current_start = 1
    # for i in 1:9
    #     action_taken[current_start:current_start+size(good_states_splitted[i], 2)-1] .= i
    #     current_start += size(good_states_splitted[i], 2)
    # end
    # good_states_loaded = hcat(good_states_splitted...)

    # println("good_states_loaded:")
    # display(good_states_loaded)
    # println("action_taken:")
    # display(action_taken)

    println("States number: $(size(states, 2))")
    # preparing working autoencoder
    NeuralNetwork.learn!(autoencoder, states, states)
    # NeuralNetwork.learn!(autoencoder, good_states_loaded, good_states_loaded)
    # action_taken_one_hot = zeros(Float32, 9, size(good_states_loaded, 2))
    # for i in 1:size(good_states_loaded, 2)
    #     action_taken_one_hot[action_taken[i], i] = 1.0
    # end
    # NeuralNetwork.learn!(autoencoder, good_states_loaded, action_taken_one_hot)
    println("Autoencoder trained")
    # println("autoencoder result: $(Environment.get_trajectory_rewards!(envs, autoencoder))")
    encoded_states = NeuralNetwork.predict(encoder, states)

    # encoded_states = NeuralNetwork.predict(encoder, good_states_loaded)

    println("Encoded states calculated")

    exemplars_ids, similarity_tree = _get_exemplars(encoded_states, encoder)
    # TEST!!!
    # encoded_states = good_states_loaded
    println("exemplars number: $(length(exemplars_ids))")
    # scatter!(encoded_states[1, exemplars_ids], encoded_states[2, exemplars_ids], legend=false, color=:red)  # Adds the exemplars to the scatter plot

    # # Saving the plot to a file, e.g., PNG format
    # savefig("scatter_plot.png")
    # println("should be scattered now")

    println("\n\n\n\nfinished\n\n\n\n")
    encoded_exemplars = encoded_states[:, exemplars_ids]

    wrapper = EnvironmentWrapperStruct(
        envs,
        encoder,
        decoder,
        autoencoder,
        (game_decoder_struct, game_decoder_kwargs),
        encoded_exemplars,
        _normalize_exemplars(encoded_exemplars),
        states,
        states[:, exemplars_ids],
        similarity_tree,
        TreeNode(nothing, nothing, 0.0, exemplars_ids),
        max_states_considered
    )

    # perfect_solution = action_taken[exemplars_ids]

    # println("exemplars ids:")
    # display(exemplars_ids)
    # println("perfect solution:")
    # display(perfect_solution)
    # println("perfect solution result: $(get_fitness(wrapper, perfect_solution))")
    
    # println("Press 'q' to exit...")
    # while true
    #     input = readline(stdin, keep=true)
    #     if input == "q"
    #         println("Exiting...")
    #         exit()
    #     end
    # end

    return wrapper
end

function copy(env_wrap::EnvironmentWrapperStruct) :: EnvironmentWrapperStruct
    envs_copy = [Environment.copy(env) for env in env_wrap._envs]
    encoder_copy = NeuralNetwork.copy(env_wrap._encoder)
    decoder_copy = NeuralNetwork.copy(env_wrap._decoder)
    autoencoder_copy = NeuralNetwork.Combined_NN([encoder_copy, decoder_copy])
    return EnvironmentWrapperStruct(
        envs_copy,
        encoder_copy,
        decoder_copy,
        autoencoder_copy,
        env_wrap._game_decoder_data,
        env_wrap._encoded_exemplars,
        env_wrap._encoded_exemplars_normalised,
        env_wrap._raw_all_states,
        env_wrap._raw_exemplars,
        env_wrap._similarity_tree,
        env_wrap._time_tree,
        env_wrap._max_states_considered
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
    envs_copies = [Environment.copy(env) for env in env_wrap._envs]
    results = Environment.get_trajectory_rewards!(envs_copies, full_NN)
    
    # previous_max_steps = env_wrap._envs[1].max_steps
    # env_wrap._envs[1].max_steps = 5_000
    # Environment.visualize!(env_wrap._envs[1], full_NN;
    #     car_image_path = raw"data\car.png",
    #     map_image_path = raw"data\map2.png",
    #     fps = 60
    # )
    # env_wrap._envs[1].max_steps = previous_max_steps
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
    states = hcat(states, env_wrap._raw_all_states)
    if size(states, 2) > env_wrap._max_states_considered
        random_columns = rand(1:size(states, 2), env_wrap._max_states_considered)
        states = states[:, random_columns]
    end
    println("States number: $(size(states, 2))")

    # We have to retrain autoencoder to cluster new states
    new_states_old_encoding = NeuralNetwork.predict(env_wrap._encoder, states)
    NeuralNetwork.learn!(env_wrap._autoencoder, states, states)
    println("Autoencoder trained")
    new_encoded_states = NeuralNetwork.predict(env_wrap._encoder, states)

    # # get new exemplars, states and newly encoded states
    new_exemplars_ids, similarity_tree = _get_exemplars(new_encoded_states, env_wrap._encoder)
    println("exemplars number: $(length(new_exemplars_ids))")
    new_exemplars = new_encoded_states[:, new_exemplars_ids]
    new_exemplars_old_encoding = new_states_old_encoding[:, new_exemplars_ids]

    # just for testing!!!!!!
    # new_states_old_encoding = NeuralNetwork.predict(env_wrap._encoder, states)
    # old_states_old_encoding = NeuralNetwork.predict(env_wrap._encoder, env_wrap._raw_all_states)
    # timestamp_string = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    # Plots.scatter(new_states_old_encoding[1, :], new_states_old_encoding[2, :], legend=false, color=:blue)  # Creates the scatter plot
    # Plots.scatter!(old_states_old_encoding[1, :], old_states_old_encoding[2, :], legend=false, color=:yellow)  # Adds the exemplars to the scatter plot
    # Plots.savefig("log/actualization_$timestamp_string.png")

    # new_states_old_encoding = hcat(new_states_old_encoding, env_wrap._encoded_exemplars)
    # env_wrap._raw_all_states = hcat(env_wrap._raw_all_states, states)

    # new_exemplars_ids, similarity_tree = _get_exemplars(new_states_old_encoding, env_wrap._encoder)
    # println("exemplars number: $(length(new_exemplars_ids))")
    # new_exemplars_old_encoding = new_states_old_encoding[:, new_exemplars_ids]
    # new_exemplars_old_encoding = hcat(new_exemplars_old_encoding, env_wrap._encoded_exemplars)
    # new_exemplars = new_exemplars_old_encoding
    # end just for testing

    # run through old encoder to get new exemplars encoded in old way, so that I can compare them with old exemplars

    new_solutions = _translate_solutions(env_wrap._encoded_exemplars, new_exemplars_old_encoding, all_solutions)
    env_wrap._encoded_exemplars = new_exemplars
    env_wrap._encoded_exemplars_normalised = _normalize_exemplars(new_exemplars)
    env_wrap._raw_all_states = states
    env_wrap._similarity_tree = similarity_tree

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
        env_wrap._encoded_exemplars_normalised,  # env_wrap._encoded_exemplars, # env_wrap._encoded_exemplars_normalised,
        genes,
        get_action_size(env_wrap)
    )
end

function translate_solutions(env_wrap_old::EnvironmentWrapperStruct, env_wrap_new::EnvironmentWrapperStruct, all_solutions::Vector{Vector{Int}}) :: Vector{Vector{Int}}
    return _translate_solutions(env_wrap_old._encoded_exemplars, NeuralNetwork.predict(env_wrap_old._encoder, env_wrap_old._raw_exemplars), all_solutions)
end

# --------------------------------------------------------------------------------------------------
# Private functions

function _collect_trajectories_states(envs::Vector{<:Environment.AbstractEnvironment}, NNs::Vector{<:NeuralNetwork.AbstractNeuralNetwork})
    trajectories_states = Vector{Array{Float32}}(undef, length(NNs))

    Threads.@threads for i in 1:length(NNs)
        envs_copy = [Environment.copy(env) for env in envs]
        nn = NNs[i]
        trajectories = Environment.get_trajectory_data!(envs_copy, nn)
        trajectories_states[i] = reduce(hcat, [trajectory.states for trajectory in trajectories])
    end

    states = reduce(hcat, trajectories_states)
    return states
end

function _normalize_exemplars(exemplars::Matrix{Float32}) :: Matrix{Float32}
    sums = sum(exemplars .^ 2, dims=1)
    exemplars_normalised = exemplars ./ sqrt.(sums)
    exemplars_normalised_transposed = exemplars_normalised'
    return exemplars_normalised_transposed
end

function _translate_solutions(old_exemplars::Matrix{Float32}, new_exemplars::Matrix{Float32}, all_solutions::Vector{Vector{Int}}) :: Vector{Vector{Int}}
    # calculate simmilarity matrix
    # distances_matrix = [vec(sum((old_exemplars .- new_exemplar).^2, dims=1)) for new_exemplar in eachcol(new_exemplars)]
    distances_matrix = Distances.pairwise(Distances.CosineDist(), old_exemplars, new_exemplars)
    new_to_old = [argmin(one_col) for one_col in eachcol(distances_matrix)]

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