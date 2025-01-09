module ContinuousStatesGroupingP3

import DataFrames
import Logging
import Printf

import ..NeuralNetwork
import ..Environment
import ..StatesGrouping
import ..AbstractOptimizerModule

include("EnvironmentWrapper.jl")
import .EnvironmentWrapper

include("IndividualModule.jl")
import .IndividualModule as Ind

mutable struct ContinuousStatesGroupingP3_Algorithm <: AbstractOptimizerModule.AbstractOptimizer
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    space_explorers_n::Int
    population::Vector{Vector{Ind.Individual}}
    best_individual::Ind.Individual
    current_env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    total_evaluations::Int
    verbose::Bool
end

function AbstractOptimizerModule.get_optimizer(::Val{:ContinuousStatesGroupingP3})
    return ContinuousStatesGroupingP3_Algorithm
end

function ContinuousStatesGroupingP3_Algorithm(;
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    env_wrapper::Dict{Symbol, <:Any},
)
    environment_type = Environment.get_environment(environment)
    environments = [(environment_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]
    visualization_env = (environment_type)(;environment_visualization_kwargs...)
    
    env_wrapper = EnvironmentWrapper.EnvironmentWrapperStruct(
        environments;
        env_wrapper...
    )

    space_explorers_n = length(environments)
    first_individual = Ind.Individual(env_wrapper)
    total_evaluations = Ind.default_FIHC!(first_individual)

    return ContinuousStatesGroupingP3_Algorithm(
        visualization_env,
        visualization_kwargs,
        space_explorers_n,
        [[first_individual]],
        first_individual,
        env_wrapper,
        total_evaluations,
        false
    )
end

function AbstractOptimizerModule.run!(p3::ContinuousStatesGroupingP3_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0) :: DataFrames.DataFrame
    EnvironmentWrapper.set_verbose!(p3.current_env_wrapper, log)
    p3.verbose = log

    # (generation, total_evaluations, best_fitness, local_individual_fitness)
    list_with_results = Vector{Tuple{Int, Int, Float64, Float64}}()

    for generation in 1:max_generations
        if p3.total_evaluations >= max_evaluations
            break
        end

        Base.GC.gc(true)  # we want to reclaim all possible memory, true means we will do full GC
        new_individual = run_new_individual!(p3)
        new_individual_fitness = Ind.get_fitness!(new_individual)
        best_individual_fitness = Ind.get_fitness!(p3.best_individual)

        best_n_distinct_individuals = get_n_best_distinct_individuals_clear_rest_memory!(p3, p3.space_explorers_n)
        p3.current_env_wrapper = EnvironmentWrapper.create_new_based_on(
            p3.current_env_wrapper,
            [
                (1.0, Ind.get_flattened_trajectories(best_n_distinct_individuals)),
                # (0.5, Ind.get_flattened_trajectories(new_individual))
            ]
        )

        if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
            Ind.visualize(p3.best_individual, p3.visualization_env, p3.visualization_kwargs)
        end

        if log
            Logging.@info "\n\n\n\n\n\nGeneration $generation\nTotal evaluations: $(p3.total_evaluations)\n" *
            Printf.@sprintf("best_fitness: %.2f\nlocal_individual_fitness: %.2f\n\n\n\n\n", best_individual_fitness, new_individual_fitness)
        end
        
        push!(
            list_with_results,
            (generation, p3.total_evaluations, best_individual_fitness, new_individual_fitness)
        )
    end

    data_frame = DataFrames.DataFrame(
        list_with_results,
        [:generation, :total_evaluations, :best_fitness, :local_individual_fitness]
    )
    return data_frame
end



# --------------------------------------------------------------------------------------------------
# logic implementation



# it will also clear trajectories of not best individuals
function get_n_best_distinct_individuals_clear_rest_memory!(p3::ContinuousStatesGroupingP3_Algorithm, n::Int) :: Vector{Ind.Individual}
    # get best individuals - from those with the same id, choose the one with the best fitness
    all_individuals = reduce(vcat, p3.population)

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
function run_new_individual!(p3::ContinuousStatesGroupingP3_Algorithm) :: Ind.Individual
    new_individual = Ind.Individual(p3.current_env_wrapper, p3.verbose)
    add_to_next_level = true
    p3.total_evaluations += Ind.default_FIHC!(new_individual)
    push!(p3.population[1], new_individual)

    # I should consider, if individuals that didnt improve through optimal mixing but only through FIHC should be added to the next level
    # currently I do so, I think it makes more sense

    i = 1
    while i <= length(p3.population)
        if p3.verbose
            Logging.@info Printf.@sprintf("\nGenes pre mixing: %s\n", Ind.get_avg_kld(new_individual, p3.population[i]))
        end

        new_individual = Ind.copy_individual(new_individual)

        old_fitness = Ind.get_fitness!(new_individual)
        p3.total_evaluations += Ind.optimal_mixing_bottom_to_top!(new_individual, p3.population[i])
        p3.total_evaluations += Ind.default_FIHC!(new_individual)

        if p3.verbose
            Logging.@info Printf.@sprintf("\nGenes post mixing: %s\n", Ind.get_avg_kld(new_individual, p3.population[i]))
        end
        new_fitness = Ind.get_fitness!(new_individual)

        if new_fitness > old_fitness
            Ind.new_level_cosideration!(new_individual)
            if i == length(p3.population)
                push!(p3.population, Vector{Ind.Individual}())
            end
            push!(p3.population[i+1], new_individual)
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



end # module ContinuousStatesGroupingP3