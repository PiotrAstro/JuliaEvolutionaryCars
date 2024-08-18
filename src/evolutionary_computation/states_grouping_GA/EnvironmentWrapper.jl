module EnvironmentWrapper

import ..NeuralNetwork
import ..Environment

import Clustering
import Statistics

export EnvironmentWrapperStruct, get_action_size, get_groups_number, get_fitness, actualize!

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct EnvironmentWrapperStruct
    _envs::Vector{<:Environment.AbstractEnvironment}
    _encoder::NeuralNetwork.AbstractNeuralNetwork
    _autoencoder::NeuralNetwork.AbstractNeuralNetwork
    _game_decoder_data::Tuple{<:Type, Dict{Symbol, Any}}
    _encoded_exemplars::Matrix{Float32}
end

# --------------------------------------------------------------------------------------------------
# Public functions

function EnvironmentWrapperStruct(
    envs::Vector{<:Environment.AbstractEnvironment},
    encoder_dict::Dict{Symbol, Any},
    decoder_dict::Dict{Symbol, Any},
    game_decoder_dict::Dict{Symbol, Any},
    initial_space_explorers_n::Int
) :: EnvironmentWrapperStruct

    encoder = NeuralNetwork.get_neural_network(encoder_dict[:name])(;encoder_dict[:kwargs]...)
    decoder = NeuralNetwork.get_neural_network(decoder_dict[:name])(;decoder_dict[:kwargs]...)
    autoencoder = NeuralNetwork.get_neural_network(decoder_dict[:name])([encoder, decoder])

    game_decoder_struct = NeuralNetwork.get_neural_network(game_decoder_dict[:name])
    game_decoder_kwargs = game_decoder_dict[:kwargs]

    # initial state space exploration
    # random NNs creation
    NNs = [
        [
            encoder,
            (game_decoder_struct)(;game_decoder_kwargs...)
        ] for _ in 1:initial_space_explorers_n
    ]
    # states collection
    states = _collect_trajectories_states(envs, NNs)
    # preparing working autoencoder
    NeuralNetwork.learn!(autoencoder, states, states)
    encoded_states = NeuralNetwork.predict(encoder, states)
    exemplars_ids = _get_exemplars(encoded_states)
    encoded_exemplars = hcat([encoded_states[:, i] for i in exemplars_ids.exemplars]...)

    return EnvironmentWrapperStruct(
        envs,
        encoder,
        autoencoder,
        (game_decoder_struct, game_decoder_kwargs),
        encoded_exemplars
    )
end

function get_action_size(env_wrap::EnvironmentWrapperStruct) :: Int
    return Environment.get_action_size(env_wrap._envs[1])
end

function get_groups_number(env_wrap::EnvironmentWrapperStruct) :: Int
    return length(env_wrap._encoded_exemplars)
end

function get_fitness(env_wrap::EnvironmentWrapperStruct, genes::Vector{Int}) :: Float64
    full_NN = _get_full_NN(env_wrap, genes)

    results = Environment.get_trajectory_rewards!(env_wrap._envs, full_NN)

    return sum(results)
end

"Actualize exemplars and autoencoder based on trajectories generated with provided genes_new_t. Then translates all_solutions to new solutions and returns them."
function actualize!(env_wrap::EnvironmentWrapperStruct, genes_new_trajectories::Vector{Vector{Int}}, all_solutions::Vector{Vector{Int}}) :: Vector{Vector{Int}}
    NNs = Vector{<:NeuralNetwork.AbstractNeuralNetwork}(undef, length(genes_new_trajectories))

    Threads.@threads for (i, solution) in enumerate(genes_new_trajectories)
        full_NN = _get_full_NN(env_wrap, solution)
        NNs[i] = full_NN
    end
    
    states = _collect_trajectories_states(env_wrap._envs, NNs)

    # We have to retrain autoencoder to cluster new states
    encoder_old_params_copy = deepcopy(NeuralNetwork.get_parameters(env_wrap._encoder))
    NeuralNetwork.learn!(env_wrap._autoencoder, states, states)
    new_encoded_states = NeuralNetwork.predict(env_wrap._encoder, states)
    encoder_new_params_copy = NeuralNetwork.get_parameters(env_wrap._encoder)

    # get new exemplars, states and newly encoded states
    new_exemplars_ids = _get_exemplars(new_encoded_states)
    new_exemplars_states = hcat([states[:, i] for i in new_exemplars_ids.exemplars]...)
    new_exemplars = hcat([new_encoded_states[:, i] for i in new_exemplars_ids.exemplars]...)

    # run through old encoder to get new exemplars encoded in old way, so that I can compare them with old exemplars
    NeuralNetwork.set_parameters!(env_wrap._encoder, encoder_old_params_copy)
    new_exemplars_old_encoded = NeuralNetwork.predict(env_wrap._encoder, new_exemplars_states)
    NeuralNetwork.set_parameters!(env_wrap._encoder, encoder_new_params_copy)

    new_solutions = _translate_solutions(env_wrap._encoded_exemplars, new_exemplars_old_encoded, all_solutions)
    env_wrap._encoded_exemplars = new_exemplars

    return new_solutions
end


# --------------------------------------------------------------------------------------------------
# Private functions

function _get_full_NN(env_wrap::EnvironmentWrapperStruct, genes::Vector{Int}) :: NeuralNetwork.AbstractNeuralNetwork
    new_game_decoder = env_wrap._game_decoder_data[1](;env_wrap._game_decoder_data[2]...)
    NeuralNetwork.train!(new_game_decoder, env_wrap._encoded_exemplars, Vector{Float32}(genes))
    full_NN = env_wrap._game_decoder_data[1](
        [
            env_wrap._encoder,
            new_game_decoder
        ]
    )

    return full_NN
end

function _collect_trajectories_states(envs::Vector{<:Environment.AbstractEnvironment}, NNs::Vector{<:NeuralNetwork.AbstractNeuralNetwork})
    trajectories_states = Vector{Array{Float32}}(undef, length(NNs))

    Threads.@threads for (i, nn) in enumerate(NNs)
        trajectories = Environment.get_trajectory_data!(envs, nn)
        trajectories_states[i] = reduce(hcat, [trajectory.states for trajectory in trajectories])
    end

    states = reduce(hcat, trajectories_states)
    return states
end

function _translate_solutions(old_exemplars::Matrix{Float32}, new_exemplars::Matrix{Float32}, all_solutions::Vector{Vector{Int}}) :: Vector{Vector{Int}}
    # calculate simmilarity matrix
    simmilarity_matrix = _cosine_simmilarity(old_exemplars, new_exemplars)

    # which old exemplar to chose for each new exemplar
    new_to_old = [argmax(@view simmilarity_matrix[:, new]) for new in 1:size(new_exemplars, 2)]

    # translate solutions
    new_solutions = Vector{Vector{Int}}()
    Threads.@threads for solution in all_solutions
        new_solution = [solution[old] for old in new_to_old]
        push!(new_solutions, new_solution)
    end

    return new_solutions
end

function _get_exemplars(encoded_states::Matrix{Float32}) :: Vector{Int}
    simmilarity_matrix = _cosine_simmilarity(encoded_states)

    # Affinity Propagation
    affinity_prop_result_median = Clustering.affinityprop(simmilarity_matrix)

    return affinity_prop_result_median.exemplars
end

function _cosine_simmilarity(states::Matrix{Float32}; diagonal_quantile_fill::Float64=0.5) :: Matrix{Float32}
    # calculate cosine simmilarity
    simmilarity_matrix = states' * states
    sqrt_sum = sqrt.(sum(states .^ 2, dims=1))
    sqrt_matrix = sqrt_sum' * sqrt_sum
    simmilarity_matrix = simmilarity_matrix ./ sqrt_matrix
    
    values_not_diagonal = [simmilarity_matrix[i, j] for i in 1:size(simmilarity_matrix, 1), j in 1:size(simmilarity_matrix, 2) if i != j]
    diagonal_fill = Statistics.quantile(values_not_diagonal, diagonal_quantile_fill)
    # along the diagonal fill median
    for i in 1:size(simmilarity_matrix, 1)
        simmilarity_matrix[i, i] = diagonal_fill
    end

    return simmilarity_matrix
end

function _cosine_simmilarity(states1::Matrix{Float32}, states2::Matrix{Float32}) :: Matrix{Float32}
    # calculate cosine simmilarity
    simmilarity_matrix = states1' * states2
    sqrt_sum1 = sqrt.(sum(states1 .^ 2, dims=1))
    sqrt_sum2 = sqrt.(sum(states2 .^ 2, dims=1))
    sqrt_matrix = sqrt_sum1' * sqrt_sum2
    simmilarity_matrix = simmilarity_matrix ./ sqrt_matrix

    return simmilarity_matrix
end

end