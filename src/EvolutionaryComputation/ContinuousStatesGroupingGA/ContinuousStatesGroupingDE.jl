module ContinuousStatesGroupingDE

import DataFrames
import Logging
import Printf
import Random
import Statistics
import LinearAlgebra

import ..NeuralNetwork
import ..Environment
import ..StatesGrouping
import ..AbstractOptimizerModule
import ..fitnesses_reduction

include("EnvironmentWrapper.jl")
import .EnvironmentWrapper

include("Individuals.jl")
import .Individuals


# --------------------------------------------------------------------------------------------------
# algorithm itself

mutable struct ContinuousStatesGroupingDE_Algorithm <: AbstractOptimizerModule.AbstractOptimizer
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    population::Vector{Individuals.Individual}
    best_individual::Individuals.Individual
    current_env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    run_statistics::Environment.RunStatistics
    individual_config::Dict{Symbol, <:Any}
    verbose::Bool
    new_individual_each_n_epochs::Int
    new_individual_genes::Symbol
end

function AbstractOptimizerModule.get_optimizer(::Val{:ContinuousStatesGroupingDE})
    return ContinuousStatesGroupingDE_Algorithm
end

function ContinuousStatesGroupingDE_Algorithm(;
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    env_wrapper::Dict{Symbol, <:Any},
    individuals_n::Int,
    individual_config::Dict{Symbol, <:Any},
    new_individual_each_n_epochs::Int,
    new_individual_genes::Symbol,
)
    run_statistics = Environment.RunStatistics()
    environment_type = Environment.get_environment(environment)
    environments = [(environment_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]
    visualization_env = (environment_type)(;environment_visualization_kwargs...)
    

    env_wrapper_struct, visualization_env = EnvironmentWrapper.EnvironmentWrapperStruct(
        environments,
        visualization_env,
        run_statistics
        ;
        env_wrapper...
    )

    individuals = [Individuals.Individual(env_wrapper_struct, individual_config) for _ in 1:individuals_n]
    # best_individual = individuals[1]
    Threads.@threads :dynamic for ind in individuals
    # for ind in individuals
        ind.env_wrapper = EnvironmentWrapper.copy(env_wrapper_struct)
        EnvironmentWrapper.random_reinitialize_exemplars!(ind.env_wrapper)
        Individuals.get_trajectories!(ind)
    end
    best_individual = individuals[argmax([Individuals.get_fitness!(ind) for ind in individuals])]

    return ContinuousStatesGroupingDE_Algorithm(
        visualization_env,
        visualization_kwargs,
        individuals,
        best_individual,
        env_wrapper_struct,
        run_statistics,
        individual_config,
        false,
        new_individual_each_n_epochs,
        new_individual_genes
    )
end

function AbstractOptimizerModule.run!(csgs::ContinuousStatesGroupingDE_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0) :: DataFrames.DataFrame

    EnvironmentWrapper.set_verbose!(csgs.current_env_wrapper, log)
    for ind in csgs.population
        ind._verbose = log
    end
    csgs.verbose = log

    quantiles = [0.25, 0.5, 0.75, 0.95]
    percentiles = (trunc(Int, 100 * quantile) for quantile in quantiles)
    percentiles_names = [Symbol("percentile_$percentile") for percentile in percentiles]
    # (generation, total_evaluations, best_fitness)
    list_with_results = Vector{Tuple}()

    for generation in 1:max_generations
        statistics = Environment.get_statistics(csgs.run_statistics)

        if statistics.total_evaluations - statistics.collected_evaluations >= max_evaluations
            break
        end

        Base.GC.gc(true)  # we want to reclaim all possible memory, true means we will do full GC

        start_time = time()

        individuals_copy_for_crossover = [Individuals.individual_copy(ind) for ind in csgs.population]

        new_env_wrap = csgs.current_env_wrapper
        new_individual = csgs.population[1]

        should_add_individual = generation % csgs.new_individual_each_n_epochs == 0
        start_from_n = ifelse(should_add_individual, 0, 1)

        # for i in start_from_n:length(csgs.population)
        Threads.@threads :dynamic for i in start_from_n:length(csgs.population)
            if i == 0
                new_env_wrap, new_individual = create_new_env_wrap_and_individual(csgs, individuals_copy_for_crossover)
            else
                Individuals.run_one_individual_generation!(csgs.population[i], individuals_copy_for_crossover)
            end
        end

        best_ind_arg = argmax(Individuals.get_fitness!.(csgs.population))
        csgs.best_individual = Individuals.individual_copy(csgs.population[best_ind_arg])

        if should_add_individual
            csgs.current_env_wrapper = new_env_wrap
            # put random individual at random place different from best individual
            random_ind = rand(collect(eachindex(csgs.population))[eachindex(csgs.population) .!= best_ind_arg])
            csgs.population[random_ind] = new_individual
        end

        end_time = time()

        # if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
        #     Ind.visualize(p3.best_individual, p3.visualization_env, p3.visualization_kwargs)
        # end

        statistics = Environment.get_statistics(csgs.run_statistics)
        distinct_evaluations = statistics.total_evaluations - statistics.collected_evaluations
        distinct_frames = statistics.total_frames - statistics.collected_frames

        best_fitness = Individuals.get_fitness!(csgs.best_individual)
        fitnesses = Individuals.get_fitness!.(csgs.population)
        quantiles_values = Statistics.quantile(fitnesses, quantiles)
        if log
            mean_fitness = Statistics.mean(fitnesses)
            elapsed_time = end_time - start_time
            Logging.@info "\n\n\n\n\n\nGeneration $generation\nTotal evaluations: $distinct_evaluations\n" *
            Printf.@sprintf("elapsed_time: %.2f s\nbest_fitness: %.2f\nmean_fitness: %.2f", elapsed_time, best_fitness, mean_fitness) *
            "quantiles:   $(join([(Printf.@sprintf "%.2f: %.2f" quantile fitness) for (quantile, fitness) in zip(quantiles, quantiles_values)], "   "))\n"
        end
        
        push!(
            list_with_results,
            (generation, distinct_evaluations, distinct_frames, best_fitness, quantiles_values...)
        )
    end

    data_frame = DataFrames.DataFrame(
        list_with_results,
        [:generation, :total_evaluations, :total_frames, :best_fitness, percentiles_names...]
    )
    return data_frame
end

function create_new_env_wrap_and_individual(csgs::ContinuousStatesGroupingDE_Algorithm, individuals_copy_for_crossover::Vector{Individuals.Individual})::Tuple{EnvironmentWrapper.EnvironmentWrapperStruct, Individuals.Individual}
    new_env_wrapper = EnvironmentWrapper.create_new_based_on(
        csgs.current_env_wrapper,
        [
            (1.0, Individuals.get_flattened_trajectories(individuals_copy_for_crossover)),
        ]
    )
    new_individual = Individuals.Individual(new_env_wrapper, csgs.individual_config, csgs.verbose)

    if csgs.new_individual_genes == :best
        Individuals.copy_genes!(new_individual, csgs.best_individual)
    elseif csgs.new_individual_genes == :rand
        # pass
    else
        throw(ArgumentError("Unknown new_individual_genes: $(csgs.new_individual_genes)"))
    end

    Individuals.get_trajectories!(new_individual)
    return new_env_wrapper, new_individual
end

end # module ContinuousStatesGroupingDE