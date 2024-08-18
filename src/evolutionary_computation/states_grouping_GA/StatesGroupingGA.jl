module StatesGroupingGA

import ..NeuralNetwork
import ..Environment

include("EnvironmentWrapper.jl")
import .EnvironmentWrapper

include("P3.jl")
import .P3

mutable struct StatesGroupingGA_Algorithm
    p3::P3.Population_Pyramid
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    visualization_env::<:Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    space_explorers_n::Int
end

function StatesGroupingGA_Algorithm(;
    nn_encoder::Dict{Symbol, Any},
    nn_autodecoder::Dict{Symbol, Any},
    nn_game_decoder::Dict{Symbol, Any},
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    space_explorers_n::Int
)
    environment_type = Environment.get_environment(environment)
    environments = [(environment_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]
    visualization_env = (environment_type)(;environment_visualization_kwargs...)
    
    env_wrapper = EnvironmentWrapper.EnvironmentWrapperStruct(
        environments,
        nn_encoder,
        nn_autodecoder,
        nn_game_decoder,
        space_explorers_n
    )

    p3 = P3.Population_Pyramid(env_wrapper)

    return StatesGroupingGA_Algorithm(
        p3,
        env_wrapper,
        visualization_env,
        visualization_kwargs,
        space_explorers_n
    )
end

function run!(algorithm::StatesGroupingGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool=true, visualize_each_n_epochs::Int=0)
    # Preprocessing data

    for generation in 1:max_generations
        # collect trajectories
        new_genes = Vector{Vector{Int}}()
        for _ in 1:algorithm.space_explorers_n
            new_individual = P3.run_new_individual(algorithm.p3)
            push!(new_genes, new_individual.genes)
        end
    end
end

end # module StatesGroupingGA