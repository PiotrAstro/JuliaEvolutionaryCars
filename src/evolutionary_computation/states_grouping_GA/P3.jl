module P3

import ..NeuralNetwork
import ..Environment

import ..EnvironmentWrapper

import Clustering
import Random

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct Individual
    genes::Vector{Int}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    _fitness::Float64
    _fitness_actual::Bool
end

mutable struct Population_Pyramid
    _population::Vector{Vector{Individual}}
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

function Population_Pyramid(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    first_individual = Individual(env_wrapper)
    return Population_Pyramid([[first_individual]], first_individual, env_wrapper)
end

function get_fitness!(individual::Individual) where E
    if !individual._fitness_actual
        individual._fitness = EnvironmentWrapper.get_fitness(individual.env_wrapper, individual.genes)
        individual._fitness_actual = true
    end

    return individual._fitness
end

function copy_individual(individual::Individual)
    return Individual(copy(individual.genes), individual.env_wrapper, individual._fitness, individual._fitness_actual)
end

function actualize_genes!(individual::Individual, new_genes::Vector{Int})
    individual.genes = new_genes
    individual._fitness_actual = false
end

# function optimal_mixing!(individual::Individual, other_individuals::Vector{Individual}, hierarchy::Clustering.)
#     new_genes = copy(individual.genes)
#     for i in 1:length(new_genes)
#         if Random.rand() < 0.5
#             new_genes[i] = other_individual.genes[i]
#         end
#     end
#     return new_genes
# end

function FIHC!(individual::Individual)
    random_order = Random.randperm(length(individual.genes))
    actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)
    for gene_index in random_order
        previous_fitness = get_fitness!(individual)
        previous_gene = individual.genes[i]
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
end

"RUns new individual, does FIHC, optimal mixing, climbing through levels and returns final individual"
function run_new_individual!(p3::Population_Pyramid) :: Vector{Int}
    new_individual = Individual(p3.env_wrapper)
    FIHC!(new_individual)
    push!(p3._population[1], new_individual)

    for level in p3._population
    end
    return new_individual.genes
end

function get_all_genes(p3::Population_Pyramid) :: Vector{Vector{Int}}
    return reduce(vcat, [
        [individual.genes for individual in level] for level in p3._population
    ])
end

function actualize_population!(p3::Population_Pyramid, changed_solutions::Vector{Vector{Int}})
    start = 1
    for level in p3._population
        stop = start - 1 + length(level)
        for (individual, changed_solution) in zip(level, changed_solutions[start:stop])
            actualize_genes!(individual, changed_solution)
        end
        start = stop + 1
    end
end

end # module P3