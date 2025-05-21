module ContinuousStatesGroupingFIHC

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

mutable struct ContinuousStatesGroupingFIHC_Algorithm <: AbstractOptimizerModule.AbstractOptimizer
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    best_individual::Individuals.Individual
    previous_individuals::Vector{Individuals.Individual}
    current_env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    run_statistics::Environment.RunStatistics
    individual_config::Dict{Symbol, <:Any}
    verbose::Bool
    new_individual_after_n_no_improvements::Int
    new_individual_genes::Symbol
end

function AbstractOptimizerModule.get_optimizer(::Val{:ContinuousStatesGroupingFIHC})
    return ContinuousStatesGroupingFIHC_Algorithm
end

function ContinuousStatesGroupingFIHC_Algorithm(;
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    env_wrapper::Dict{Symbol, <:Any},
    individual_config::Dict{Symbol, <:Any},
    new_individual_after_n_no_improvements::Int,
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
    best_individual = Individuals.Individual(env_wrapper_struct, individual_config)

    return ContinuousStatesGroupingFIHC_Algorithm(
        visualization_env,
        visualization_kwargs,
        best_individual,
        Vector{Individuals.Individual}(),
        env_wrapper_struct,
        run_statistics,
        individual_config,
        false,
        new_individual_after_n_no_improvements::Int,
        new_individual_genes
    )
end

function AbstractOptimizerModule.run!(csgs::ContinuousStatesGroupingFIHC_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0) :: DataFrames.DataFrame

    EnvironmentWrapper.set_verbose!(csgs.current_env_wrapper, log)
    csgs.best_individual._verbose = log
    current_individual = Individuals.individual_copy(csgs.best_individual)
    no_improvement_n = 0
    csgs.verbose = log

    list_with_results = Vector{Tuple}()

    for generation in 1:max_generations
        statistics = Environment.get_statistics(csgs.run_statistics)

        if statistics.total_evaluations - statistics.collected_evaluations >= max_evaluations
            break
        end

        Base.GC.gc(true)  # we want to reclaim all possible memory, true means we will do full GC

        start_time = time()
        prev_fitness = Individuals.get_fitness!(current_individual)
        Individuals.FIHC_generation!(current_individual)
        current_fitness = Individuals.get_fitness!(current_individual)
        if current_fitness > prev_fitness
            no_improvement_n = 0
        else
            no_improvement_n += 1
        end

        if no_improvement_n >= csgs.new_individual_after_n_no_improvements
            new_env_wrapper, new_individual = create_new_env_wrap_and_individual(csgs, current_individual)
            current_individual = new_individual
            csgs.current_env_wrapper = new_env_wrapper
            no_improvement_n = 0
        end
        end_time = time()

        if current_fitness > Individuals.get_fitness!(csgs.best_individual)
            csgs.best_individual = Individuals.individual_copy(current_individual)
        end

        if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
            Environment.visualize!(csgs.visualization_env, Individuals.get_nn(current_individual))
        end

        statistics = Environment.get_statistics(csgs.run_statistics)
        distinct_evaluations = statistics.total_evaluations - statistics.collected_evaluations
        distinct_frames = statistics.total_frames - statistics.collected_frames

        best_fitness = Individuals.get_fitness!(csgs.best_individual)
        if log
            elapsed_time = end_time - start_time
            Logging.@info "\n\n\n\n\n\nGeneration $generation\nTotal evaluations: $distinct_evaluations\n" *
            Printf.@sprintf("elapsed_time: %.2f s\nbest_fitness: %.2f\ncurrent_fitness: %.2f\n", elapsed_time, best_fitness, current_fitness)
        end
        
        push!(
            list_with_results,
            (generation, distinct_evaluations, distinct_frames, best_fitness, current_fitness)
        )
    end

    data_frame = DataFrames.DataFrame(
        list_with_results,
        [:generation, :total_evaluations, :total_frames, :best_fitness, :current_fitness]
    )
    return data_frame
end

function create_new_env_wrap_and_individual(csgs::ContinuousStatesGroupingFIHC_Algorithm, current_ind::Individuals.Individual)::Tuple{EnvironmentWrapper.EnvironmentWrapperStruct, Individuals.Individual}
    push!(csgs.previous_individuals, Individuals.individual_copy(current_ind))
    new_env_wrapper = EnvironmentWrapper.create_new_based_on(
        csgs.current_env_wrapper,
        [
            (1.0, Individuals.get_flattened_trajectories(csgs.previous_individuals)),
        ]
    )
    new_individual = Individuals.Individual(new_env_wrapper, csgs.individual_config, csgs.verbose)

    if csgs.new_individual_genes == :best
        Individuals.copy_genes!(new_individual, csgs.best_individual)
    elseif csgs.new_individual_genes == :rand
        # pass
    elseif csgs.new_individual_genes == :current
        Individuals.copy_genes!(new_individual, current_ind)
    else
        throw(ArgumentError("Unknown new_individual_genes: $(csgs.new_individual_genes)"))
    end

    Individuals.get_trajectories!(new_individual)
    return new_env_wrapper, new_individual
end

end # module ContinuousStatesGroupingDE