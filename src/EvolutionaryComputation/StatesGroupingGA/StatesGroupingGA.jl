module StatesGroupingGA

import ..NeuralNetwork
import ..Environment

include("EnvironmentWrapper.jl")
import .EnvironmentWrapper

include("IndividualModule.jl")
import .IndividualModule

# --------------------------------------------------------------------------------------------------
# Ga methods

include("MutationOnly.jl")
import .MutationOnly

include("P3Levels.jl")
import .P3Levels

include("NormalGA.jl")
import .NormalGA

mutable struct StatesGroupingGA_Algorithm
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    space_explorers_n::Int
end

function StatesGroupingGA_Algorithm(;
    nn_encoder::Dict{Symbol, Any},
    nn_autodecoder::Dict{Symbol, Any},
    nn_game_decoder::Dict{Symbol, Any},
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    space_explorers_n::Int,
    max_states_considered::Int,
    n_clusters::Int,
    fuzzy_logic_of_n_closest::Int
)
    environment_type = Environment.get_environment(environment)
    environments = [(environment_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]
    visualization_env = (environment_type)(;environment_visualization_kwargs...)
    
    env_wrapper = EnvironmentWrapper.EnvironmentWrapperStruct(
        environments,
        nn_encoder,
        nn_autodecoder,
        nn_game_decoder,
        space_explorers_n,
        max_states_considered,
        n_clusters,
        fuzzy_logic_of_n_closest
    )

    return StatesGroupingGA_Algorithm(
        env_wrapper,
        visualization_env,
        visualization_kwargs,
        space_explorers_n
    )
end

function run!(algorithm::StatesGroupingGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool=true, visualize_each_n_epochs::Int=0)
    P3Levels.run!(algorithm.env_wrapper, algorithm.visualization_env, algorithm.visualization_kwargs, max_generations, algorithm.space_explorers_n)
    # MutationOnly.run!(algorithm.env_wrapper, algorithm.visualization_env, algorithm.visualization_kwargs, max_generations, algorithm.space_explorers_n)
    # NormalGA.run!(algorithm.env_wrapper, algorithm.visualization_env, algorithm.visualization_kwargs, max_generations, algorithm.space_explorers_n)
end

end # module StatesGroupingGA