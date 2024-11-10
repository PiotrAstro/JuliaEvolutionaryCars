# it has different state of environment on each level (next levels are more actualised)
# therefore in general I do not have to actualise fitnesses of individuals - only the one that moves from one level to another, so I can FIHC it
module MutationOnly

import ..IndividualModule as Ind
import ..NeuralNetwork
import ..Environment
import ..EnvironmentWrapper

import Random
import Plots
import Dates
import Statistics

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct MutationOnlyStruct
    population::Vector{Ind.Individual}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    best_individual::Ind.Individual
end

# --------------------------------------------------------------------------------------------------
# Public functions


function MutationOnlyStruct(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    individuals = [Ind.Individual(env_wrapper) for _ in 1:100]
    return MutationOnlyStruct(individuals, env_wrapper, individuals[1])
end

function get_n_best_individuals_genes(mutation_struct::MutationOnlyStruct, n::Int) :: Vector{Vector{Int}}
    all_genes = [individual.genes for individual in sort(mutation_struct.population, by=Ind.get_fitness!, rev=true)]
    best_genes = all_genes[1:min(n, length(all_genes))]
    return best_genes
end

function generation!(mutation_struct::MutationOnlyStruct)
    new_individuals = Vector{Ind.Individual}(undef, length(mutation_struct.population))
    Threads.@threads for i in 1:length(mutation_struct.population)
        Ind.get_fitness!(mutation_struct.population[i])
        new_individuals[i] = Ind.copy_individual(mutation_struct.population[i])
        Ind.mutate_top_to_bottom!(new_individuals[i], 0.05)
        Ind.get_fitness!(new_individuals[i])
    end

    combined_population = vcat(mutation_struct.population, new_individuals)
    mutation_struct.population = sort(combined_population, by=Ind.get_fitness!, rev=true)[1:length(mutation_struct.population)]
    
    if Ind.get_fitness!(mutation_struct.population[1]) > Ind.get_fitness!(mutation_struct.best_individual)
        Ind.FIHC_top_to_bottom!(mutation_struct.population[1])
        mutation_struct.best_individual = mutation_struct.population[1]
    end

    fitnesses = [Ind.get_fitness!(individual) for individual in mutation_struct.population]
    quantiles = [0.25, 0.5, 0.75, 0.95]
    quantile_fitnesses = Statistics.quantile(fitnesses, quantiles)
    quantile_text = join(["$(quantiles[i]): $(quantile_fitnesses[i])" for i in 1:length(quantiles)], "    ")
    println("Best fitness: $(Ind.get_fitness!(mutation_struct.best_individual))")
    println("Quantiles: $quantile_text\n\n")
end

# --------------------------------------------------------------------------------------------------

function run!(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, visualization_env::Environment.AbstractEnvironment, visualization_kwargs::Dict{Symbol, Any}, max_generations::Int, space_explorers_n::Int)
    mutation_struct = MutationOnlyStruct(env_wrapper)
    # Preprocessing data
    best_ever_fitness = -Inf
    best_ever_fitness_environment_wrapper_version = 0
    previous_generation_change = 0

    current_env_wrapper_version = 0
    for generation in 1:max_generations
        println("Generation global: $generation")
        println("best_ever_fitness: $best_ever_fitness   best_ever_fitness_environment_wrapper_version: $best_ever_fitness_environment_wrapper_version")
        println("Generation local: $(generation - previous_generation_change)   current_env_wrapper_version: $current_env_wrapper_version")
        generation!(mutation_struct)

        if Ind.get_fitness!(mutation_struct.best_individual) > best_ever_fitness
            best_ever_fitness = Ind.get_fitness!(mutation_struct.best_individual)
            best_ever_fitness_environment_wrapper_version = current_env_wrapper_version
        end

        if (generation - previous_generation_change) % (250 * 2^current_env_wrapper_version) == 0
            current_env_wrapper_version += 1
            env_wrap = EnvironmentWrapper.copy(mutation_struct.env_wrapper)
            best_individuals = get_n_best_individuals_genes(mutation_struct, space_explorers_n)
            EnvironmentWrapper.actualize!(env_wrap, best_individuals, best_individuals)
            mutation_struct = MutationOnlyStruct(env_wrap)
            previous_generation_change = generation
        end
    end
end



end # module P3