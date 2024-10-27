module StatesGroupingGA

import ..NeuralNetwork
import ..Environment
import ..ClusteringHML

include("EnvironmentWrapper.jl")
import .EnvironmentWrapper

include("P3.jl")
import .P3

include("P3Levels.jl")
import .P3Levels

include("NormalGA.jl")
import .NormalGA

mutable struct StatesGroupingGA_Algorithm
    # p3::P3.Population_Pyramid
    p3Levels::P3Levels.Population_Pyramid
    # ga::NormalGA.GA_Struct
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    visualization_env::Environment.AbstractEnvironment
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
    space_explorers_n::Int,
    max_states_considered::Int
)
    environment_type = Environment.get_environment(environment)
    environments = [(environment_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]
    visualization_env = (environment_type)(;environment_visualization_kwargs...)
    
    env_wrapper = EnvironmentWrapper.EnvironmentWrapperStruct(
        environments,
        nn_encoder,
        nn_autodecoder,
        nn_game_decoder,
        space_explorers_n,
        max_states_considered
    )

    # p3 = P3.Population_Pyramid(env_wrapper)
    p3Levels = P3Levels.Population_Pyramid(env_wrapper)
    # ga = NormalGA.GA_Struct(env_wrapper)

    return StatesGroupingGA_Algorithm(
        # p3,
        p3Levels,
        # ga,
        env_wrapper,
        visualization_env,
        visualization_kwargs,
        space_explorers_n
    )
end

function run!(algorithm::StatesGroupingGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool=true, visualize_each_n_epochs::Int=0)
    # Preprocessing data

    for generation in 1:max_generations
        P3Levels.run_new_individual!(algorithm.p3Levels)


        # collect trajectories
        # new_genes_vector = Vector{Vector{<:Int}}()
        # for _ in 1:algorithm.space_explorers_n
        #     new_genes = P3.run_new_individual!(algorithm.p3)
        #     push!(new_genes_vector, new_genes)
        # end
        # P3.run_new_individual!(algorithm.p3)


        # @time NormalGA.generation!(algorithm.ga)
        # # nn = EnvironmentWrapper.get_full_NN(algorithm.env_wrapper, NormalGA.get_best_genes(algorithm.ga))
        # # Environment.visualize!(algorithm.visualization_env, nn; algorithm.visualization_kwargs...)

        # if generation % 1 == 0
        #     best_n_genes = NormalGA.get_best_n_genes(algorithm.ga, algorithm.space_explorers_n)
        #     all_solutions = NormalGA.get_all_genes(algorithm.ga)

        #     translated_solutions = EnvironmentWrapper.actualize!(algorithm.env_wrapper, best_n_genes, all_solutions)
        #     NormalGA.actualize_population!(algorithm.ga, translated_solutions)
        # end
    end
end

end # module StatesGroupingGA