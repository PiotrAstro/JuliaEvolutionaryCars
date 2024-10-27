include("../NeuralNetwork/NeuralNetwork.jl")
include("../constants.jl")

import .NeuralNetwork

import JLD
import Plots


const ACTION_NUM = 9
const DATA_DIR = "log/_evolutionary_mutate_population/"
const COLOURS = [:red, :blue, :green, :yellow, :purple, :orange, :brown, :pink, :grey]

function plot_and_save(name::String, action_set::Vector{Int})
    action_vectors = [Vector{Int}(undef, 0) for _ in 1:ACTION_NUM]
    for i in eachindex(action_set)
        push!(action_vectors[action_set[i]], i)
    end

    Plots.scatter([0], [0], legend=false, marker=:x, size=(2000, 2000), markerstrokewidth=0)
    for i in 1:ACTION_NUM
        Plots.scatter!(encoded_states[1, action_vectors[i]], encoded_states[2, action_vectors[i]], legend=false, marker=:o, markerstrokewidth=0, color=COLOURS[i])
    end

    Plots.savefig("$(DATA_DIR)$(name).png")
end

# create NN
encoder_dict = CONSTANTS_DICT[:StatesGroupingGA][:nn_encoder]
decoder_dict = CONSTANTS_DICT[:StatesGroupingGA][:nn_autodecoder]

encoder = NeuralNetwork.get_neural_network(encoder_dict[:name])(;encoder_dict[:kwargs]...)
decoder = NeuralNetwork.get_neural_network(decoder_dict[:name])(;decoder_dict[:kwargs]...)
autoencoder = NeuralNetwork.Combined_NN([encoder, decoder])

# load data
states = JLD.load("$(DATA_DIR)states.jld")["states"]
actions = JLD.load("$(DATA_DIR)actions.jld")

# train autoencoder
NeuralNetwork.learn!(autoencoder, states, states)

encoded_states = NeuralNetwork.predict(encoder, states)

for (individual_name, actions_individual) in actions
    plot_and_save(individual_name, actions_individual)
end