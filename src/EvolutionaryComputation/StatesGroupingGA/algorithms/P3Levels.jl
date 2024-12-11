# it has different state of environment on each level (next levels are more actualised)
# therefore in general I do not have to actualise fitnesses of individuals - only the one that moves from one level to another, so I can FIHC it
module P3Levels

import ..IndividualModule as Ind
import ..NeuralNetwork
import ..Environment
import ..EnvironmentWrapper

import Random
import Plots
import Dates
import Logging
import Printf
import DataFrames

# --------------------------------------------------------------------------------------------------
# Structs

struct Level 
    individuals::Vector{Ind.Individual}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
end

mutable struct Population_Pyramid
    population::Vector{Level}
    # _dependencies_matrices::Vector{Array{Int, 4}}
    best_individual::Ind.Individual
    verbose::Bool
end

# --------------------------------------------------------------------------------------------------
# Public functions

function Population_Pyramid(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, verbose::Bool = true) :: Population_Pyramid
    first_individual = Ind.Individual(env_wrapper, verbose)
    Ind.FIHC_top_to_bottom!(first_individual)

    return Population_Pyramid([Level([first_individual], env_wrapper)], first_individual, verbose)
end

function get_n_best_individuals_genes(p3::Population_Pyramid, n::Int) :: Vector{Vector{Int}}
    all_genes = [individual.genes for individual in sort(reduce(vcat, [level.individuals for level in p3.population]), by=Ind.get_fitness!, rev=true)]
    best_genes = all_genes[1:min(n, length(all_genes))]
    return best_genes
end


"Runs new individual, does FIHC, optimal mixing, climbing through levels and returns final individual"
function run_new_individual!(p3::Population_Pyramid)
    if p3.verbose
        Logging.@info Printf.@sprintf("best individual fitness: %.2f\n", Ind.get_fitness!(p3.best_individual))
    end
    new_individual = Ind.Individual(p3.population[1].env_wrapper, p3.verbose)
    Ind.FIHC_top_to_bottom!(new_individual)
    # save_decision_plot(new_individual)
    add_to_next_level = true

    i = 1
    while i <= length(p3.population)
        new_individual = Ind.copy_individual(new_individual)
        if i > 1
            Ind.FIHC_top_to_bottom!(new_individual)
        end

        if p3.verbose
            Logging.@info Printf.@sprintf("\nGenes pre mixing: %s\n", Ind.get_same_genes_percent(new_individual, p3.population[i].individuals))
        end

        if add_to_next_level
            push!(p3.population[i].individuals, new_individual)
        end
        new_individual = Ind.copy_individual(new_individual)

        old_fitness = Ind.get_fitness!(new_individual)
        Ind.optimal_mixing_bottom_to_top!(new_individual, p3.population[i].individuals)

        if p3.verbose
            Logging.@info Printf.@sprintf("\nGenes post mixing: %s\n", Ind.get_same_genes_percent(new_individual, p3.population[i].individuals))
        end
        new_fitness = Ind.get_fitness!(new_individual)

        if Ind.get_fitness!(p3.best_individual) < new_fitness
            p3.best_individual = Ind.copy_individual(new_individual)
            if p3.verbose
                Logging.@info Printf.@sprintf("\n\nnew best individual fitness: %.2f\n\n", Ind.get_fitness!(p3.best_individual))
            end
        end

        if new_fitness > old_fitness
            add_to_next_level = true

            if i == length(p3.population)
                # new_env_wrapper = EnvironmentWrapper.copy(p3.population[i].env_wrapper)
                # # actually I do not want to actualise any genes, i just want to create new empty level
                # genes = [individual.genes for individual in p3.population[i].individuals]
                # EnvironmentWrapper.actualize!(new_env_wrapper, genes, genes)
                # push!(p3.population, Level(Vector{Individual}(undef, 0), new_env_wrapper))
                push!(p3.population, Level(Vector{Ind.Individual}(undef, 0), p3.population[i].env_wrapper))
            end
        else
            add_to_next_level = false
        end

        i += 1
    end
end


# --------------------------------------------------------------------------------------------------

function run!(
        env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
        ;
        visualization_env::Environment.AbstractEnvironment,
        visualization_kwargs::Dict{Symbol, Any},
        max_generations::Int,
        space_explorers_n::Int,
        max_evaluations::Int,
        log::Bool=true,
        visualize_each_n_epochs::Int=0
    )
    EnvironmentWrapper.set_verbose!(env_wrapper, false)
    p3 = Population_Pyramid(env_wrapper, log)
    run_new_individual!(p3)
    # Preprocessing data
    best_ever_fitness = -Inf
    best_ever_fitness_environment_wrapper_version = 0

    # (generation_global, best_fitness_global, generation_local, best_fitness_local)
    list_with_results = Vector{Tuple{Int, Float64, Int, Float64}}()

    current_env_wrapper_version = 0
    generation_this_level = 0
    for generation in 1:max_generations
        generation_this_level += 1
        run_new_individual!(p3)
        if Ind.get_fitness!(p3.best_individual) > best_ever_fitness
            best_ever_fitness = Ind.get_fitness!(p3.best_individual)
            best_ever_fitness_environment_wrapper_version = current_env_wrapper_version
        end

        if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
            Ind.visualize(p3.best_individual, visualization_env, visualization_kwargs)
        end

        if log
            Logging.@info "\n\n\n\n\n\nGeneration $generation\n" *
            Printf.@sprintf("Generation local: %d   current_env_wrapper_version: %d   current_best_fitness: %.2f\n", generation_this_level, current_env_wrapper_version, Ind.get_fitness!(p3.best_individual)) *
            Printf.@sprintf("Generation global: %d   best_ever_fitness: %.2f   best_ever_fitness_environment_wrapper_version: %d\n\n\n\n\n\n", generation, best_ever_fitness, best_ever_fitness_environment_wrapper_version)
        end

        if generation_this_level % (2 * 2^current_env_wrapper_version) == 0
            current_env_wrapper_version += 1
            env_wrap = EnvironmentWrapper.copy(p3.population[1].env_wrapper)
            best_individuals = get_n_best_individuals_genes(p3, space_explorers_n)
            EnvironmentWrapper.actualize!(env_wrap, best_individuals, best_individuals)
            p3 = P3Levels.Population_Pyramid(env_wrap, p3.verbose)
            generation_this_level = 0
        end

        push!(
            list_with_results,
            (generation, best_ever_fitness, generation_this_level, Ind.get_fitness!(p3.best_individual))
        )
    end

    data_frame = DataFrames.DataFrame(
        list_with_results,
        [:generation_global, :best_fitness_global, :generation_local, :best_fitness_local]
    )
    return data_frame
end



end # module P3