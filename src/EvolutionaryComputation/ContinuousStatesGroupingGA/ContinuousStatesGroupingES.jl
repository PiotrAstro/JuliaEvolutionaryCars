module ContinuousStatesGroupingES

import DataFrames
import Logging
import Printf
import Random
import Statistics
import LinearAlgebra

import ..NeuralNetwork
import ..Environment
import ..ES
import ..StatesGrouping
import ..AbstractOptimizerModule
import ..fitnesses_reduction


include("EnvironmentWrapper.jl")
import .EnvironmentWrapper

include("Individuals.jl")
import .Individuals


# --------------------------------------------------------------------------------------------------
# algorithm itself

mutable struct ContinuousStatesGroupingES_Algorithm <: AbstractOptimizerModule.AbstractOptimizer
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    previous_individuals::Vector{Individuals.Individual}
    current_env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    run_statistics::Environment.RunStatistics
    individual_config::Dict{Symbol, <:Any}
    verbose::Bool
    es_type
    es_kwargs::Dict
    new_es_after_n_iterations::Int
end

function AbstractOptimizerModule.get_optimizer(::Val{:ContinuousStatesGroupingES})
    return ContinuousStatesGroupingES_Algorithm
end

function ContinuousStatesGroupingES_Algorithm(;
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    env_wrapper::Dict{Symbol, <:Any},
    individual_config::Dict{Symbol, <:Any},
    es_type::Symbol,
    es_kwargs::Dict{Symbol, <:Any},
    new_es_after_n_iterations::Int,
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

    # for i in 1:20
    #     ind_tmp = Individuals.Individual(env_wrapper_struct, individual_config)
    #     Environment.visualize!(visualization_env, Individuals.get_nn(ind_tmp))
    # end
    # throw("dddd")

    return ContinuousStatesGroupingES_Algorithm(
        visualization_env,
        visualization_kwargs,
        Vector{Individuals.Individual}(),
        env_wrapper_struct,
        run_statistics,
        individual_config,
        false,
        ES.get_es(es_type),
        es_kwargs,
        new_es_after_n_iterations,
    )
end

function AbstractOptimizerModule.run!(csgs::ContinuousStatesGroupingES_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0) :: DataFrames.DataFrame

    EnvironmentWrapper.set_verbose!(csgs.current_env_wrapper, log)
    es = csgs.es_type(
        zeros(Float32, EnvironmentWrapper.get_action_size(csgs.current_env_wrapper), EnvironmentWrapper.get_groups_number(csgs.current_env_wrapper))
        ;
        csgs.es_kwargs...
    )
    best_fitness = -Inf
    csgs.verbose = log

    list_with_results = Vector{Tuple}()

    for generation in 1:max_generations
        statistics = Environment.get_statistics(csgs.run_statistics)

        if statistics.total_evaluations - statistics.collected_evaluations >= max_evaluations
            break
        end

        Base.GC.gc(true)  # we want to reclaim all possible memory, true means we will do full GC

        if generation % csgs.new_es_after_n_iterations == 0
            means = ES.get_mean(es)
            # this one should be created as new one, cause we will add it to the previous individuals pool
            mean_individual = Individuals.Individual(csgs.current_env_wrapper, csgs.individual_config, csgs.verbose)
            Individuals.set_genes!(mean_individual, means)
            new_env_wrap, new_individual = create_new_env_wrap_and_individual(csgs, mean_individual)
            new_means = new_individual.genes  # dont have to copy it, it should be copied inside es constructor  Base.copy(new_individual.genes)
            es = csgs.es_type(new_means; csgs.es_kwargs...)
        end

        start_time = time()
        means = ES.get_mean(es)
        solutions = ES.generate_solutions!(es)
        individuals = Vector{Individuals.Individual}(undef, length(solutions) + 1)

        Threads.@threads :dynamic for i in 0:length(solutions)
        # for i in 0:length(solutions)
            ind_local = Individuals.Individual(csgs.current_env_wrapper, csgs.individual_config, csgs.verbose)
            if i == 0
                Individuals.set_genes!(ind_local, means)
            else
                Individuals.set_genes!(ind_local, solutions[i])
            end
            Individuals.get_fitness!(ind_local)
            individuals[i + 1] = ind_local
        end
        fitnesses = [Individuals.get_fitness!(ind) for ind in individuals[2:end]]
        ES.update!(es, solutions, fitnesses)
        end_time = time()

        iteration_best_fitness = -Inf
        iteration_base_fitness = Individuals.get_fitness!(individuals[1])
        for ind in individuals
            local_fitness = Individuals.get_fitness!(ind)
            if local_fitness > iteration_best_fitness
                iteration_best_fitness = local_fitness
            end
        end

        if iteration_best_fitness > best_fitness
            best_fitness = iteration_best_fitness
        end

        if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
            Environment.visualize!(csgs.visualization_env, Individuals.get_nn(individuals[1]))
        end

        statistics = Environment.get_statistics(csgs.run_statistics)
        distinct_evaluations = statistics.total_evaluations - statistics.collected_evaluations
        distinct_frames = statistics.total_frames - statistics.collected_frames

        if log
            elapsed_time = end_time - start_time
            Logging.@info "\n\n\n\n\n\nGeneration $generation\nTotal evaluations: $distinct_evaluations\n" *
            Printf.@sprintf("elapsed_time: %.2f s\nbest_fitness: %.2f\ncurrent_base_fitness (means): %.2f\niteration_best: %.2f\n", elapsed_time, best_fitness, iteration_base_fitness, iteration_best_fitness)
        end
        
        push!(
            list_with_results,
            (generation, distinct_evaluations, distinct_frames, best_fitness, iteration_base_fitness, iteration_best_fitness)
        )
    end

    data_frame = DataFrames.DataFrame(
        list_with_results,
        [:generation, :total_evaluations, :total_frames, :best_fitness, :base_fitness, :iteration_best_fitness]
    )
    return data_frame
end

function create_new_env_wrap_and_individual(csgs::ContinuousStatesGroupingES_Algorithm, current_ind::Individuals.Individual)::Tuple{EnvironmentWrapper.EnvironmentWrapperStruct, Individuals.Individual}
    Individuals.get_trajectories!(current_ind)
    push!(csgs.previous_individuals, Individuals.individual_copy(current_ind))
    new_env_wrapper = EnvironmentWrapper.create_new_based_on(
        csgs.current_env_wrapper,
        [
            (1.0, Individuals.get_flattened_trajectories(csgs.previous_individuals)),
        ]
    )
    new_individual = Individuals.Individual(new_env_wrapper, csgs.individual_config, csgs.verbose)
    Individuals.copy_genes!(new_individual, current_ind)
    return new_env_wrapper, new_individual
end

end # module ContinuousStatesGroupingDE