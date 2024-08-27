module NormalGA

import ..NeuralNetwork
import ..Environment

import ..EnvironmentWrapper

import Clustering
import Random
import StatsBase

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct Individual
    genes::Vector{Int}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    _fitness::Float64
    _fitness_actual::Bool
end

mutable struct GA_Struct
    _population::Vector{Individual}
    # _dependencies_matrices::Vector{Array{Int, 4}}
    best_individual::Individual
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
end

# --------------------------------------------------------------------------------------------------
# Public functions

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    genes = Random.rand(1:EnvironmentWrapper.get_action_size(env_wrapper), EnvironmentWrapper.get_groups_number(env_wrapper))
    fitness = -Inf
    fitness_actual = false
    return Individual(genes, env_wrapper, fitness, fitness_actual)
end

function GA_Struct(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    individuals = [Individual(env_wrapper) for _ in 1:100]
    return GA_Struct(individuals, individuals[1], env_wrapper)
end

function get_fitness!(individual::Individual) :: Float64
    if !individual._fitness_actual
        individual._fitness = EnvironmentWrapper.get_fitness(individual.env_wrapper, individual.genes)
        individual._fitness_actual = true
    end

    return individual._fitness
end

function copy_individual(individual::Individual)
    return Individual(copy(individual.genes), individual.env_wrapper, individual._fitness, individual._fitness_actual)
end

function crossover(individual1::Individual, individual2::Individual) :: Tuple{Individual, Individual}
    new_individual1 = copy_individual(individual1)
    new_individual2 = copy_individual(individual2)
    for i in 1:length(new_individual1.genes)
        if Random.rand() < 0.5
            new_individual1.genes[i] = individual2.genes[i]
            new_individual2.genes[i] = individual1.genes[i]
        end
    end

    new_individual1._fitness_actual = false
    new_individual2._fitness_actual = false

    return (new_individual1, new_individual2)
end

function mutate!(individual::Individual, probability::Float64)
    action_size = EnvironmentWrapper.get_action_size(individual.env_wrapper)
    for i in 1:length(individual.genes)
        if Random.rand() < probability
            individual.genes[i] = Random.rand(1:action_size)
        end
    end
    individual._fitness_actual = false
end

function tournament_selection(ga::GA_Struct) :: Tuple{Individual, Individual}
    # select 4 random distinct individuals
    individuals = StatsBase.sample(ga._population, 4, replace=false)
    individual1 = individuals[1]
    if Random.rand() < 0.5
        if Random.rand() < 0.5
            individual1 = individuals[2]
        end
    elseif get_fitness!(individuals[2]) > get_fitness!(individual1)
        individual1 = individuals[2]
    end

    individual2 = individuals[3]
    if Random.rand() < 0.5
        if Random.rand() < 0.5
            individual2 = individuals[4]
        end
    elseif get_fitness!(individuals[4]) > get_fitness!(individual2)
        individual2 = individuals[4]
    end

    return (individual1, individual2)
end

function _check_new_best!(ga::GA_Struct)
    for individual in ga._population
        if get_fitness!(individual) > get_fitness!(ga.best_individual)
            FIHC!(individual)
            ga.best_individual = copy_individual(individual)
        end
    end
end

function generation!(ga::GA_Struct)
    for i in 1:5
        ga._population[i] = Individual(ga.env_wrapper)
    end
    new_population = Vector{Individual}(undef, length(ga._population))
    Threads.@threads for i in 1:2:length(ga._population)
        individual1, individual2 = tournament_selection(ga)
        new_individual1, new_individual2 = crossover(individual1, individual2)
        mutate!(new_individual1, 0.05)
        mutate!(new_individual2, 0.05)

        original = get_fitness!(individual1) > get_fitness!(individual2) ? individual1 : individual2
        new_individual = get_fitness!(new_individual1) > get_fitness!(new_individual2) ? new_individual1 : new_individual2
        new_population[i] = original
        new_population[i + 1] = new_individual

        # println("parents fitness: ", get_fitness!(individual1), " ", get_fitness!(individual2), " new fitness: ", get_fitness!(new_individual1), " ", get_fitness!(new_individual2))
    end

    ga._population = new_population
    _check_new_best!(ga)

    println("Best fitness: ", get_fitness!(ga.best_individual))
end

function FIHC!(individual::Individual)
    print("\npre FIHC fitness: ", get_fitness!(individual))

    random_order = Random.randperm(length(individual.genes))
    actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)
    for gene_index in random_order
        previous_fitness = get_fitness!(individual)
        previous_gene = individual.genes[gene_index]
        random_loci_order = [i for i in Random.randperm(actions_number) if i != previous_gene]
        for loci in random_loci_order
            individual.genes[gene_index] = loci
            individual._fitness_actual = false
            new_fitness = get_fitness!(individual)
            if new_fitness > previous_fitness
                break
            else
                individual.genes[gene_index] = previous_gene
                individual._fitness = previous_fitness
            end
        end
    end

    print("\tpost FIHC fitness: $(get_fitness!(individual))\n")
end

function get_best_genes(ga::GA_Struct) :: Vector{Int}
    return ga.best_individual.genes
end

function get_best_n_genes(ga::GA_Struct, n::Int) :: Vector{Vector{Int}}
    return [individual.genes for individual in sort(ga._population, by=get_fitness!, rev=true)[1:n]]
end

function get_all_genes(ga::GA_Struct) :: Vector{Vector{Int}}
    return [individual.genes for individual in ga._population]
end

function actualize_population!(ga::GA_Struct, changed_solutions::Vector{Vector{Int}})
    println("population_length: ", length(ga._population), " changed_solutions_length: ", length(changed_solutions))
    for (individual, changed_solution) in zip(ga._population, changed_solutions)
        actualize_genes!(individual, changed_solution)
    end

    ga.best_individual = sort(ga._population, by=get_fitness!, rev=true)[1]
end

function actualize_genes!(individual::Individual, new_genes::Vector{Int})
    println("Actualizing genes, previous genes length: ", length(individual.genes), " new genes length: ", length(new_genes))
    individual.genes = new_genes
    individual._fitness_actual = false

    println("previous fitness: ", individual._fitness, " new fitness: ", get_fitness!(individual))
end

# # function optimal_mixing!(individual::Individual, other_individuals::Vector{Individual}, hierarchy::Clustering.)
# #     new_genes = copy(individual.genes)
# #     for i in 1:length(new_genes)
# #         if Random.rand() < 0.5
# #             new_genes[i] = other_individual.genes[i]
# #         end
# #     end
# #     return new_genes
# # end

# function FIHC!(individual::Individual)
#     println("pre FIHC fitness: ", get_fitness!(individual))

#     random_order = Random.randperm(length(individual.genes))
#     actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)
#     for gene_index in random_order
#         previous_fitness = get_fitness!(individual)
#         previous_gene = individual.genes[gene_index]
#         random_loci_order = [i for i in Random.randperm(actions_number) if i != previous_gene]
#         for loci in random_loci_order
#             individual.genes[gene_index] = loci
#             individual._fitness_actual = false
#             new_fitness = get_fitness!(individual)
#             if new_fitness > previous_fitness
#                 break
#             else
#                 individual.genes[gene_index] = previous_gene
#                 individual._fitness = previous_fitness
#             end
#         end
#     end

#     print("post FIHC fitness: ", get_fitness!(individual))
# end

# "RUns new individual, does FIHC, optimal mixing, climbing through levels and returns final individual"
# function run_new_individual!(p3::Population_Pyramid) :: Vector{Int}
#     new_individual = Individual(p3.env_wrapper)
#     FIHC!(new_individual)
#     push!(p3._population[1], new_individual)

#     for level in p3._population
#     end
#     return new_individual.genes
# end

# function get_all_genes(p3::Population_Pyramid) :: Vector{Vector{Int}}
#     return reduce(vcat, [
#         [individual.genes for individual in level] for level in p3._population
#     ])
# end

# function actualize_population!(p3::Population_Pyramid, changed_solutions::Vector{Vector{Int}})
#     start = 1
#     for level in p3._population
#         stop = start - 1 + length(level)
#         for (individual, changed_solution) in zip(level, changed_solutions[start:stop])
#             actualize_genes!(individual, changed_solution)
#         end
#         start = stop + 1
#     end
# end

end # module NormalGA