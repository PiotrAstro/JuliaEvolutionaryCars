module NormalGA

import ..IndividualModule as Ind
import ..NeuralNetwork
import ..Environment
import ..EnvironmentWrapper

import Clustering
import Random
import Statistics
import StatsBase
import Plots
import Dates

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct GA_Struct
    _population::Vector{Ind.Individual}
    # _dependencies_matrices::Vector{Array{Int, 4}}
    best_individual::Ind.Individual
    const env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
end

# --------------------------------------------------------------------------------------------------
# Public functions

function GA_Struct(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    individuals = [Ind.Individual(env_wrapper) for _ in 1:200]
    # for individual in individuals
    Threads.@threads for i in 1:length(individuals)
        Ind.get_fitness!(individuals[i])
    end
    ga_struct = GA_Struct(individuals, individuals[1], env_wrapper)
    _check_new_best!(ga_struct)
    return ga_struct
end

function tournament_selection(ga::GA_Struct) :: Tuple{Ind.Individual, Ind.Individual}
    # select 4 random distinct individuals
    individuals = StatsBase.sample(ga._population, 4, replace=false)
    individual1 = individuals[1]
    if Random.rand() < 0.5
        if Random.rand() < 0.5
            individual1 = individuals[2]
        end
    elseif Ind.get_fitness!(individuals[2]) > Ind.get_fitness!(individual1)
        individual1 = individuals[2]
    end

    individual2 = individuals[3]
    if Random.rand() < 0.5
        if Random.rand() < 0.5
            individual2 = individuals[4]
        end
    elseif Ind.get_fitness!(individuals[4]) > Ind.get_fitness!(individual2)
        individual2 = individuals[4]
    end

    return (individual1, individual2)
end

function _check_new_best!(ga::GA_Struct)
    for individual in ga._population
        if Ind.get_fitness!(individual) > Ind.get_fitness!(ga.best_individual)
            Ind.FIHC_top_to_bottom!(individual)
            ga.best_individual = Ind.copy_individual(individual)

            # if get_fitness!(individual) > 480.0
            # Ind.save_decision_plot(individual)
            # end
        end
    end
end

function generation!(ga::GA_Struct)
    for i in 1:1
        ga._population[i] = Ind.Individual(ga.env_wrapper)
    end
    random_perm_population = Random.shuffle(ga._population)
    new_population = Vector{Ind.Individual}(undef, length(ga._population))
    Threads.@threads for i in 1:2:length(ga._population)
    # for i in 1:2:length(ga._population)
        # individual1, individual2 = tournament_selection(ga)
        # new_individual1, new_individual2 = crossover(individual1, individual2)
        # mutate!(new_individual1, 0.05)
        # mutate!(new_individual2, 0.05)

        # original = get_fitness!(individual1) > get_fitness!(individual2) ? individual1 : individual2
        # new_individual = get_fitness!(new_individual1) > get_fitness!(new_individual2) ? new_individual1 : new_individual2
        # new_population[i] = original
        # new_population[i + 1] = new_individual

        # individual1, individual2 = tournament_selection(ga)
        individual1, individual2 = random_perm_population[i], random_perm_population[i + 1]
        original = Ind.get_fitness!(individual1) > Ind.get_fitness!(individual2) ? individual1 : individual2
        new_individual = Ind.optimal_mixing_top_to_bottom_2_individuals(individual1, individual2)
        new_population[i] = original
        new_population[i + 1] = new_individual

        # println("parents fitness: ", get_fitness!(individual1), " ", get_fitness!(individual2), " new fitness: ", get_fitness!(new_individual1), " ", get_fitness!(new_individual2))
    end

    ga._population = new_population
    _check_new_best!(ga)

    fitnesses = [Ind.get_fitness!(individual) for individual in ga._population]
    quantiles = [0.25, 0.5, 0.75, 0.95]
    quantile_fitnesses = Statistics.quantile(fitnesses, quantiles)
    quantile_text = join(["$(quantiles[i]): $(quantile_fitnesses[i])" for i in 1:length(quantiles)], "    ")
    println("Best fitness: $(Ind.get_fitness!(ga.best_individual))")
    println("Quantiles: $quantile_text\n\n")
end

function get_best_genes(ga::GA_Struct) :: Vector{Int}
    return ga.best_individual.genes
end

function get_best_n_genes(ga::GA_Struct, n::Int) :: Vector{Vector{Int}}
    return [individual.genes for individual in sort(ga._population, by=Ind.get_fitness!, rev=true)[1:n]]
end

function get_all_genes(ga::GA_Struct) :: Vector{Vector{Int}}
    return [individual.genes for individual in ga._population]
end

function actualize_population!(ga::GA_Struct, changed_solutions::Vector{Vector{Int}})
    println("population_length: ", length(ga._population), " changed_solutions_length: ", length(changed_solutions))
    for (individual, changed_solution) in zip(ga._population, changed_solutions)
        Ind.actualize_genes!(individual, changed_solution)
    end

    ga.best_individual = sort(ga._population, by=Ind.get_fitness!, rev=true)[1]
end

function run!(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, visualization_env::Environment.AbstractEnvironment, visualization_kwargs::Dict{Symbol, Any}, max_generations::Int, space_explorers_n::Int)
    ga_struct = GA_Struct(env_wrapper)
    # Preprocessing data
    best_ever_fitness = -Inf
    best_ever_fitness_environment_wrapper_version = 0
    previous_generation_change = 0

    current_env_wrapper_version = 0
    for generation in 1:max_generations
        println("Generation global: $generation")
        println("best_ever_fitness: $best_ever_fitness   best_ever_fitness_environment_wrapper_version: $best_ever_fitness_environment_wrapper_version")
        println("Generation local: $(generation - previous_generation_change)   current_env_wrapper_version: $current_env_wrapper_version")
        generation!(ga_struct)

        if Ind.get_fitness!(ga_struct.best_individual) > best_ever_fitness
            best_ever_fitness = Ind.get_fitness!(ga_struct.best_individual)
            best_ever_fitness_environment_wrapper_version = current_env_wrapper_version
        end

        if (generation - previous_generation_change) % (250 * 2^current_env_wrapper_version) == 0
            current_env_wrapper_version += 1
            env_wrap = EnvironmentWrapper.copy(ga_struct.env_wrapper)
            best_individuals = get_best_n_genes(ga_struct, space_explorers_n)
            EnvironmentWrapper.actualize!(env_wrap, best_individuals, best_individuals)
            mutation_struct = GA_Struct(env_wrap)
            previous_generation_change = generation
        end
    end
end

end # module NormalGA