module NormalGA

import ..IndividualModule as Ind
import ..NeuralNetwork
import ..Environment
import ..EnvironmentWrapper

import Clustering
import Random
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
    individuals = [Ind.Individual(env_wrapper) for _ in 1:100]
    for individual in individuals
        Ind.get_fitness!(individual)
    end
    return GA_Struct(individuals, individuals[1], env_wrapper)
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
            Ind.save_decision_plot(individual)
            # end
        end
    end
end

function generation!(ga::GA_Struct)
    for i in 1:1
        ga._population[i] = Ind.Individual(ga.env_wrapper)
    end
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

        individual1, individual2 = tournament_selection(ga)
        original = Ind.get_fitness!(individual1) > Ind.get_fitness!(individual2) ? individual1 : individual2
        new_individual = Ind.optimal_mixing(individual1, individual2)
        new_population[i] = original
        new_population[i + 1] = new_individual

        # println("parents fitness: ", get_fitness!(individual1), " ", get_fitness!(individual2), " new fitness: ", get_fitness!(new_individual1), " ", get_fitness!(new_individual2))
    end

    ga._population = new_population
    _check_new_best!(ga)

    println("Best fitness: ", Ind.get_fitness!(ga.best_individual))
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

end # module NormalGA