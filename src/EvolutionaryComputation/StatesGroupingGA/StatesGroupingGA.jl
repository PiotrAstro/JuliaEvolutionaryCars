module StatesGroupingGA

import DataFrames

import ..NeuralNetwork
import ..Environment
import ..StatesGrouping
import ..AbstractOptimizerModule



include("EnvironmentWrapper.jl")
import .EnvironmentWrapper

include("IndividualModule.jl")
import .IndividualModule

# --------------------------------------------------------------------------------------------------
# Ga methods

# include("algorithms/MutationOnly.jl")
# import .MutationOnly

include("algorithms/P3Levels.jl")
import .P3Levels

# include("algorithms/NormalGA.jl")
# import .NormalGA

mutable struct StatesGroupingGA_Algorithm <: AbstractOptimizerModule.AbstractOptimizer
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    space_explorers_n::Int
end

function StatesGroupingGA_Algorithm(;
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

    return StatesGroupingGA_Algorithm(
        env_wrapper,
        visualization_env,
        visualization_kwargs,
        space_explorers_n
    )
end

function AbstractOptimizerModule.get_optimizer(::Val{:StatesGroupingGA})
    return StatesGroupingGA_Algorithm
end

function AbstractOptimizerModule.run!(algorithm::StatesGroupingGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0) :: DataFrames.DataFrame
    return P3Levels.run!(algorithm.env_wrapper;
        visualization_env=algorithm.visualization_env,
        visualization_kwargs=algorithm.visualization_kwargs,
        max_generations=max_generations,
        space_explorers_n=algorithm.space_explorers_n,
        max_evaluations=max_evaluations,
        log=log,
        visualize_each_n_epochs=visualize_each_n_epochs
    )
    # MutationOnly.run!(algorithm.env_wrapper, algorithm.visualization_env, algorithm.visualization_kwargs, max_generations, algorithm.space_explorers_n)
    # NormalGA.run!(algorithm.env_wrapper, algorithm.visualization_env, algorithm.visualization_kwargs, max_generations, algorithm.space_explorers_n)
end

end # module StatesGroupingGA