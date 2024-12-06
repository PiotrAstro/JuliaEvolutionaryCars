module StatesGroupingGA

import ..NeuralNetwork
import ..Environment

include("EnvironmentWrapper/EnvironmentWrapper.jl")
import .EnvironmentWrapper

include("IndividualModule.jl")
import .IndividualModule

# --------------------------------------------------------------------------------------------------
# Ga methods

include("algorithms/MutationOnly.jl")
import .MutationOnly

include("algorithms/P3Levels.jl")
import .P3Levels

include("algorithms/NormalGA.jl")
import .NormalGA

mutable struct StatesGroupingGA_Algorithm
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    space_explorers_n::Int
end

function StatesGroupingGA_Algorithm(;
    nn_encoder::Dict{Symbol, Any},
    nn_decoder::Dict{Symbol, Any},
    nn_autoencoder::Dict{Symbol, <:Any},
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
        nn_decoder,
        nn_autoencoder,
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

function run!(algorithm::StatesGroupingGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0)
    P3Levels.run!(algorithm.env_wrapper;
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