module ContinuousStatesGroupingSimpleGA

import DataFrames
import Logging
import Printf

import ..NeuralNetwork
import ..Environment
import ..StatesGrouping
import ..AbstractOptimizerModule

include("EnvironmentWrapper.jl")
import .EnvironmentWrapper

# --------------------------------------------------------------------------------------------------
# individuals

mutable struct Individual
    genes::Matrix{Float32}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    _trajectories::Vector{<:Environment.Trajectory}
    _trajectories_actual::Bool
    _fitness::Float64
    _fitness_actual::Bool
    _verbose::Bool
end

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, verbose=false)
    genes = EnvironmentWrapper.new_genes(env_wrapper)
    return Individual(
        genes,
        env_wrapper,
        Vector{Environment.Trajectory}(undef, 0),
        false,
        -Inf64,
        false,
        verbose
    )
end

function individual_copy(ind::Individual)
    Individual(
        Base.copy(ind.genes),
        ind.env_wrapper,
        ind._trajectories,
        ind._trajectories_actual,
        ind._fitness,
        ind._fitness_actual,
        ind._verbose
    )
end

function set_non_actual(ind::Individual)
    ind._fitness_actual = false
    ind._trajectories_actual = false
end

function get_flattened_trajectories(individuals::Vector{Individual})::Vector{<:Environment.Trajectory}
    return vcat([individual._trajectories for individual in individuals]...)
end

function get_fitness!(individual::Individual)::Float64
    if !individual._fitness_actual
        individual._fitness = EnvironmentWrapper.get_fitness(individual.env_wrapper, individual.genes)
        individual._fitness_actual = true
    end

    return individual._fitness
end

function get_trajectories!(individual::Individual)::Float64
    if !individual._trajectories_actual
        individual._trajectories = EnvironmentWrapper.get_trajectories(individual.env_wrapper, individual.genes)
        individual._trajectories_actual = true
        individual._fitness_actual = true
        individual._fitness = sum([tra.result for tra in individual._trajectories])
    end

    return individual._fitness
end

function FIHC_crossover!(other_individuals::Vector{Individual})
    
end

# --------------------------------------------------------------------------------------------------
# algorith itself

mutable struct ContinuousStatesGroupingSimpleGA_Algorithm <: AbstractOptimizerModule.AbstractOptimizer
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    population::Vector{Individual}
    best_individual::Individual
    current_env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    total_evaluations::Int
    verbose::Bool
end

function AbstractOptimizerModule.get_optimizer(::Val{:ContinuousStatesGroupingSimpleGA})
    return ContinuousStatesGroupingSimpleGA_Algorithm
end

function ContinuousStatesGroupingSimpleGA_Algorithm(;
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

function AbstractOptimizerModule.run!(p3::ContinuousStatesGroupingSimpleGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0) :: DataFrames.DataFrame
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


end # module ContinuousStatesGroupingP3