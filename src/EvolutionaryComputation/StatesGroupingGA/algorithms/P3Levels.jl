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
end

mutable struct Population_Pyramid
    population::Vector{Level}
    # _dependencies_matrices::Vector{Array{Int, 4}}
    best_individual::Ind.Individual
    current_env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    verbose::Bool
end

# --------------------------------------------------------------------------------------------------
# Public functions

function Population_Pyramid(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, verbose::Bool = true) :: Population_Pyramid
    first_individual = Ind.Individual(env_wrapper, verbose)
    Ind.FIHC_top_to_bottom!(first_individual)

    return Population_Pyramid([Level([first_individual])], first_individual, env_wrapper, verbose)
end

# it will also clear trajectories of not best individuals
function get_n_best_distinct_individuals_clear_rest_memory!(p3::Population_Pyramid, n::Int) :: Vector{Ind.Individual}
    # get best individuals - from those with the same id, choose the one with the best fitness
    all_individuals = reduce(vcat, [level.individuals for level in p3.population])

    best_individuals = Dict{Int, Ind.Individual}()
    for individual in all_individuals
        id = Ind.get_id_track(individual)
        other_individual = get!(best_individuals, id, individual)
        if Ind.get_fitness!(individual) > Ind.get_fitness!(other_individual)

            # !!! clearing memory
            Ind.clear_trajectory_memory!(other_individual)
            best_individuals[id] = individual
        end
    end
    all_distinct_individuals = [individual for individual in values(best_individuals)]
    all_distinct_individuals_sorted = sort(all_distinct_individuals, by=Ind.get_fitness!, rev=true)
    take_n_individuals = min(n, length(all_distinct_individuals_sorted))

    # !!! clearing memory
    for individual in all_distinct_individuals_sorted[(take_n_individuals+1):end]
        Ind.clear_trajectory_memory!(individual)
    end

    best_individuals = all_distinct_individuals_sorted[1:take_n_individuals]
    return best_individuals
end

"Runs new individual, does FIHC, optimal mixing, climbing through levels and returns final individual"
function run_new_individual!(p3::Population_Pyramid) :: Ind.Individual
    new_individual = Ind.Individual(p3.current_env_wrapper, p3.verbose)
    add_to_next_level = true

    i = 1
    while i <= length(p3.population)
        Ind.FIHC_top_to_bottom!(new_individual)

        if p3.verbose
            Logging.@info Printf.@sprintf("\nGenes pre mixing: %s\n", Ind.get_same_genes_percent(new_individual, p3.population[i].individuals))
        end

        if add_to_next_level
            Ind.new_level_cosideration!(new_individual)
            push!(p3.population[i].individuals, new_individual)
        end
        new_individual = Ind.copy_individual(new_individual)

        old_fitness = Ind.get_fitness!(new_individual)
        Ind.optimal_mixing_bottom_to_top!(new_individual, p3.population[i].individuals)

        if p3.verbose
            Logging.@info Printf.@sprintf("\nGenes post mixing: %s\n", Ind.get_same_genes_percent(new_individual, p3.population[i].individuals))
        end
        new_fitness = Ind.get_fitness!(new_individual)

        if new_fitness > old_fitness
            add_to_next_level = true
            if i == length(p3.population)
                push!(p3.population, Level(Vector{Ind.Individual}(undef, 0)))
            end
        else
            add_to_next_level = false
        end

        if Ind.get_fitness!(p3.best_individual) < new_fitness
            p3.best_individual = Ind.copy_individual(new_individual)
            if p3.verbose
                Logging.@info Printf.@sprintf("\n\nnew best individual fitness: %.2f\n\n", Ind.get_fitness!(p3.best_individual))
            end
        end

        i += 1
    end

    return new_individual
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
    EnvironmentWrapper.set_verbose!(env_wrapper, log)
    p3 = Population_Pyramid(env_wrapper, log)

    # (generation, best_fitness, local_individual_fitness)
    list_with_results = Vector{Tuple{Int, Float64, Float64}}()

    for generation in 1:max_generations
        Base.GC.gc(true)  # we want to reclaim all possible memory, true means we will do full GC
        new_individual = run_new_individual!(p3)
        new_individual_fitness = Ind.get_fitness!(new_individual)
        best_individual_fitness = Ind.get_fitness!(p3.best_individual)

        # before creating new env wrapper, we will clean memory of the current one, so that we will have enough memory for new one
        EnvironmentWrapper.clean_memory!(p3.current_env_wrapper)

        best_n_distinct_individuals = get_n_best_distinct_individuals_clear_rest_memory!(p3, space_explorers_n)
        p3.current_env_wrapper = EnvironmentWrapper.create_new_based_on(
            p3.current_env_wrapper,
            [
                (1.0, Ind.get_flattened_trajectories(best_n_distinct_individuals)),
                # (0.5, Ind.get_flattened_trajectories(new_individual))
            ]
        )

        if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
            Ind.visualize(p3.best_individual, visualization_env, visualization_kwargs)
        end

        if log
            Logging.@info "\n\n\n\n\n\nGeneration $generation\n" *
            Printf.@sprintf("best_fitness: %.2f\nlocal_individual_fitness: %.2f\n\n\n\n\n", best_individual_fitness, new_individual_fitness)
        end
        
        push!(
            list_with_results,
            (generation, best_individual_fitness, new_individual_fitness)
        )
    end

    data_frame = DataFrames.DataFrame(
        list_with_results,
        [:generation, :best_fitness, :local_individual_fitness]
    )
    return data_frame
end



end # module P3