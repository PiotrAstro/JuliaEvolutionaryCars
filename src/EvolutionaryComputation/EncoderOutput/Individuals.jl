module Individuals

import ..Environment
import ..NeuralNetwork
import ..StatesGrouping
import ..EnvironmentWrapper
import ..fitnesses_reduction

export Individual, IndividualConfig, get_fitness!, get_trajectories!, individual_copy, get_flattened_trajectories, get_nn, set_genes!

# --------------------------------------------------------------------------------------------------
# individuals

@kwdef struct IndividualConfig
    fitnesses_reduction::Symbol = :sum
end

mutable struct Individual
    genes::Matrix{Float32}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    config::IndividualConfig
    _trajectories::Vector{<:Environment.Trajectory}
    _trajectories_actual::Bool
    _fitness::Float64
    _fitness_actual::Bool
    _verbose::Bool
end

function Individual(
        env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct,
        individual_config::Dict
        ;
        genes::Union{Matrix{Float32}, Nothing}=nothing,
        verbose::Bool=false
    )::Individual
    config = IndividualConfig(;individual_config...)

    if isnothing(genes)
        genes = randn(Float32, EnvironmentWrapper.get_latent_size(env_wrapper), EnvironmentWrapper.get_action_size(env_wrapper))
    end
    
    return Individual(
        genes,
        env_wrapper,
        config,
        Vector{Environment.Trajectory}(undef, 0),
        false,
        -Inf64,
        false,
        verbose
    )
end

function set_genes!(ind::Individual, genes::Matrix{Float32})
    ind.genes = genes
    ind._fitness_actual = false
    ind._trajectories_actual = false
end

function individual_copy(ind::Individual)
    Individual(
        Base.copy(ind.genes),
        ind.env_wrapper,
        ind.config,
        ind._trajectories,
        ind._trajectories_actual,
        ind._fitness,
        ind._fitness_actual,
        ind._verbose
    )
end

function get_nn(ind::Individual)
    return EnvironmentWrapper.get_full_NN(ind.env_wrapper, ind.genes)
end

function get_flattened_trajectories(individuals::Vector{Individual})::Vector{<:Environment.Trajectory}
    return reduce(vcat, [individual._trajectories for individual in individuals])
end

function get_fitness!(individual::Individual)::Float64
    if !individual._fitness_actual
        reduction_mehod = individual.config.fitnesses_reduction
        individual._fitness = fitnesses_reduction(reduction_mehod, EnvironmentWrapper.get_fitnesses(individual.env_wrapper, individual.genes))
        individual._fitness_actual = true
    end

    return individual._fitness
end

function get_trajectories!(individual::Individual)::Float64
    if !individual._trajectories_actual
        individual._trajectories = EnvironmentWrapper.get_trajectories(individual.env_wrapper, individual.genes)
        individual._trajectories_actual = true
        individual._fitness_actual = true
        reduction_mehod = individual.config.fitnesses_reduction
        individual._fitness = fitnesses_reduction(reduction_mehod, [tra.rewards_sum for tra in individual._trajectories])
    end

    return individual._fitness
end

end