module ContinuousStatesGrouping

import ..NeuralNetwork
import ..Environment

import PyCall
genieclust = PyCall.pyimport("genieclust")
import Random
import StatsBase

export ContinuousStatesGroupingAlgorithm

include("_group_states.jl")
include("_individual.jl")

# Algorithm itself

mutable struct ContinuousStatesGroupingAlgorithm
    # const population_levels::Vector{Vector{Individual}}
    const population::Vector{Individual}
    const visualization_kwargs::Dict{Symbol, Any}
    const visualization_environment::Environment.AbstractEnvironment
    const environments::Vector{<:Environment.AbstractEnvironment}
    best_individual::Individual
    const n_threads::Int
end

function ContinuousStatesGroupingAlgorithm(;
    n_threads::Int,
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    neural_network_data::Dict{Symbol, Any}
) :: ContinuousStatesGroupingAlgorithm
    nn_type = NeuralNetwork.get_neural_network(neural_network_data[:name])
    env_type = Environment.get_environment(environment)
    
    visualization_environment = (env_type)(;environment_visualization_kwargs...)
    environments = [(env_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]

    best_individual = Individual(nn_type, neural_network_data[:kwargs], environments)

    # for p3
    # local_search!(best_individual)
    # population_levels = [[best_individual]]
    # return ContinuousStatesGroupingAlgorithm(
    #     population_levels,
    #     visualization_kwargs,
    #     visualization_environment,
    #     environments,
    #     best_individual,
    #     n_threads > 0 ? n_threads : Threads.nthreads()
    # )

    # for normal ga:
    population = [Individual(nn_type, neural_network_data[:kwargs], environments) for _ in 1:100]
    return ContinuousStatesGroupingAlgorithm(
        population,
        visualization_kwargs,
        visualization_environment,
        environments,
        best_individual,
        n_threads > 0 ? n_threads : Threads.nthreads()
    )
end

function run!(algo::ContinuousStatesGroupingAlgorithm; max_generations::Int, max_evaluations::Int, log::Bool=true, visualize_each_n_epochs::Int=0)
    for generation in 1:max_generations
        random_individual = Individual(algo.best_individual.neural_network_type, algo.best_individual.neural_network_kwargs, algo.environments)
        algo.population[rand(1:length(algo.population))] = random_individual

        random_permutation = Random.randperm(length(algo.population))
        Threads.@threads for i in 1:2:length(random_permutation)
            individual_1 = algo.population[random_permutation[i]]
            individual_2 = algo.population[random_permutation[i+1]]
            new_individual = crossover(individual_1, individual_2)
            new_individual_index = get_fitness(individual_1) > get_fitness(individual_2) ? i + 1 : i
            algo.population[random_permutation[new_individual_index]] = new_individual
        end

        best_individual_population = algo.population[argmax([get_fitness(individual) for individual in algo.population])]

        if get_fitness(best_individual_population) > get_fitness(algo.best_individual)
            # local_search!(best_individual_population)
            algo.best_individual = best_individual_population
        end

        if log
            println("Generation: $generation")
            println("Best fitness: $(get_fitness(algo.best_individual))")
            quantiles = StatsBase.quantile([get_fitness(individual) for individual in algo.population], [0.25, 0.5, 0.75])
            println("Quantiles: $quantiles")
        end
    end
end

# p3 version
# function run!(algo::ContinuousStatesGroupingAlgorithm; max_generations::Int, max_evaluations::Int, log::Bool=true, visualize_each_n_epochs::Int=0)
#     while true
#         new_individual = Individual(algo.best_individual.neural_network_type, algo.best_individual.neural_network_kwargs, algo.environments)
#         local_search!(new_individual)
#         push!(algo.population_levels[1], new_individual)

#         for (i, level) in enumerate(algo.population_levels)
#             old_fitness = get_fitness(new_individual)
#             for individual in level[Random.randperm(length(level))]
#                 new_individual_2 = crossover(new_individual, individual)
#                 if get_fitness(new_individual_2) > get_fitness(new_individual)
#                     new_individual = new_individual_2
#                 end
#             end
#             local_search!(new_individual)

#             if get_fitness(new_individual) > old_fitness
#                 if get_fitness(new_individual) > get_fitness(algo.best_individual)
#                     algo.best_individual = new_individual
#                 end

#                 if i == length(algo.population_levels)
#                     push!(algo.population_levels, [new_individual])
#                     break
#                 else
#                     push!(algo.population_levels[i+1], new_individual)
#                 end
#             end
#         end
#     end
# end

end