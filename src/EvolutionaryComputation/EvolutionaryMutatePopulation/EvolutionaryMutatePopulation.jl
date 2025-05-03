module EvolutionaryMutatePopulaiton
    
import ..NeuralNetwork
import ..Environment
import ..AbstractOptimizerModule
import ..fitnesses_reduction

import Statistics as St
import Printf as Pf
import JLD
import Random
import DataFrames

# tmp
import Lux

export EvolutionaryMutatePopulationAlgorithm

#--------------------------------------------------------------------------------------------------------
# protected

mutable struct Individual
    neural_network::N where N<:NeuralNetwork.AbstractTrainableAgentNeuralNetwork
    _fitness::Float64
    _is_fitness_calculated::Bool
    environments::Vector{<:Environment.AbstractEnvironment}
    run_statistics::Environment.RunStatistics
    fitnesses_reduction_method::Symbol
end

function Individual(
        neural_network_type::Type{N},
        neural_network_kwargs::Dict{Symbol, Any},
        envs::Vector{T},
        run_statistics::Environment.RunStatistics,
        fitnesses_reduction_method::Symbol
    ) where {T<:Environment.AbstractEnvironment, N<:NeuralNetwork.AbstractTrainableAgentNeuralNetwork}
    return Individual(
        (neural_network_type)(;neural_network_kwargs...),
        0.0,
        false,
        envs,
        run_statistics,
        fitnesses_reduction_method
    )
end

function get_fitness(ind::Individual)::Float64
    if !ind._is_fitness_calculated
        ind._fitness = fitnesses_reduction(ind.fitnesses_reduction_method, Environment.get_trajectory_rewards!(ind.environments, ind.neural_network; run_statistics=ind.run_statistics, reset=true))
        ind._is_fitness_calculated = true
    end
    return ind._fitness
end


"Returns new individual with mutated neural network and the same environments."
function mutate(ind::Individual, mutation_rate::Float64) :: Individual
    get_fitness(ind)
    new_individual = copy(ind)

    params = NeuralNetwork.copy_parameters(new_individual.neural_network)
    mutate_traverse!(params, mutation_rate)
    NeuralNetwork.set_parameters!(new_individual.neural_network, params)

    # NeuralNetwork.set_parameters!(new_individual.neural_network, params)
    new_individual._is_fitness_calculated = false
    get_fitness(new_individual)
    return new_individual
end

function mutate_traverse!(params::NamedTuple, mutation_rate::Float64)
    for value in values(params)
        if value isa NamedTuple
            mutate_traverse!(value, mutation_rate)
        elseif value isa AbstractArray
            value .+= randn(Float32, size(value)) .* mutation_rate
        elseif value isa Number
            value .+= randn(Float32) * mutation_rate
        end
    end
end

function copy(ind::Individual) :: Individual
    new_individual = Individual(
        NeuralNetwork.copy(ind.neural_network),
        ind._fitness,
        ind._is_fitness_calculated,
        [Environment.copy(env) for env in ind.environments],
        ind.run_statistics,
        ind.fitnesses_reduction_method
    )
    return new_individual
end


#--------------------------------------------------------------------------------------------------------
# public

mutable struct EvolutionaryMutatePopulationAlgorithm <: AbstractOptimizerModule.AbstractOptimizer
    population::Vector{Individual}
    population_size::Int
    mutation_rate::Float64
    const visualization_kwargs::Dict{Symbol, Any}
    const visualization_environment::Environment.AbstractEnvironment
    best_individual::Individual
    run_statistics::Environment.RunStatistics
end

function AbstractOptimizerModule.get_optimizer(::Val{:Evolutionary_Mutate_Population})
    return EvolutionaryMutatePopulationAlgorithm
end

function EvolutionaryMutatePopulationAlgorithm(;
    population_size::Int,
    mutation_rate::Float64,
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    neural_network_data::Dict{Symbol, Any},
    fitnesses_reduction_method::Symbol=:sum,
    environment_norm::Union{Nothing, Symbol}=nothing,
    environment_norm_kwargs::Union{Dict, Nothing}=nothing,
    environment_norm_individuals::Int=30,
    environment_norm_activation::Symbol=:none,
) :: EvolutionaryMutatePopulationAlgorithm
    nn_type = NeuralNetwork.get_neural_network(neural_network_data[:name])
    env_type = Environment.get_environment(environment)
    envs = [(env_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]
    
    visualization_environment = (env_type)(;environment_visualization_kwargs...)
    run_statistics = Environment.RunStatistics()

    if !isnothing(environment_norm)
        env_norm_type = Environment.get_environment(environment_norm)
        random_nns = [NeuralNetwork.Random_NN(Environment.get_action_size(envs[1]), environment_norm_activation) for _ in 1:environment_norm_individuals]
        trajectories = [Environment.get_trajectory_data!(envs, nn; run_statistics=run_statistics, reset=true) for nn in random_nns]
        flat_trajectories = vcat(trajectories...)
        states = [trajectory.states for trajectory in flat_trajectories]
        asseq = Environment.get_ASSEQ(envs[1])(states)

        if isnothing(environment_norm_kwargs)
            env_norms = Environment.get_norm_data(env_norm_type, asseq)
        else
            env_norms = Environment.get_norm_data(env_norm_type, asseq; environment_norm_kwargs...)
        end
        envs = [env_norm_type(env, env_norms) for env in envs]
        visualization_environment = env_norm_type(visualization_environment, env_norms)
    end

    population = Vector([
        Individual(
            nn_type,
            neural_network_data[:kwargs],
            [Environment.copy(env) for env in envs],
            run_statistics,
            fitnesses_reduction_method
    ) for _ in 1:population_size])

    return EvolutionaryMutatePopulationAlgorithm(
        population,
        population_size,
        mutation_rate,
        visualization_kwargs,
        visualization_environment,
        copy(population[1]),
        run_statistics
    )
end

function AbstractOptimizerModule.run!(algo::EvolutionaryMutatePopulationAlgorithm; max_generations::Int, max_evaluations::Int, log::Bool=true, visualize_each_n_epochs::Int=0)
    quantiles = [0.25, 0.5, 0.75, 0.95]
    percentiles = (trunc(Int, 100 * quantile) for quantile in quantiles)
    percentiles_names = [Symbol("percentile_$percentile") for percentile in percentiles]
    # (generation, total_evaluations, best_fitness)
    list_with_results = Vector{Tuple}()

    for generation in 1:max_generations
        statistics = Environment.get_statistics(algo.run_statistics)

        if statistics.total_evaluations - statistics.collected_evaluations >= max_evaluations
            break
        end

        time_start = time()

        new_individuals = Vector(algo.population)
        Threads.@threads for i in eachindex(new_individuals)
        # for i in eachindex(new_individuals)
            # new_individuals[i] = @time mutate(new_individuals[i], algo.mutation_rate)
            new_individuals[i] = mutate(new_individuals[i], algo.mutation_rate)
        end

        append!(algo.population, new_individuals)

        sorted = sort(algo.population, by=get_fitness, rev=true)
        algo.population = sorted[1:algo.population_size]
        time_end = time()

        statistics = Environment.get_statistics(algo.run_statistics)
        distinct_evaluations = statistics.total_evaluations - statistics.collected_evaluations
        distinct_frames = statistics.total_frames - statistics.collected_frames

        if get_fitness(algo.population[1]) > get_fitness(algo.best_individual)
            algo.best_individual = copy(algo.population[1])
        end

        quantiles_values = St.quantile(get_fitness.(algo.population), quantiles)

        if log    
            elapsed_time = time_end - time_start
            one_mutation_ratio = Threads.nthreads() * elapsed_time / length(algo.population)
            Pf.@printf "Generation: %i, time: %.3f, threads: %i, calculated: %i, time*threads/calc: %.3f\n" generation elapsed_time Threads.nthreads() length(algo.population_size) one_mutation_ratio
            Pf.@printf "best: %.2f\tmean: %.2f\n" get_fitness(algo.best_individual) St.mean(get_fitness.(algo.population))
            println("quantiles:\t$(join([(Pf.@sprintf "%.2f: %.2f" quantile fitness) for (quantile, fitness) in zip(quantiles, quantiles_values)], "\t"))")
            print("\n\n\n")
        end

        if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
            Environment.visualize!(algo.visualization_environment, algo.best_individual.neural_network; algo.visualization_kwargs...)
        end

        push!(
            list_with_results,
            (generation, distinct_evaluations, distinct_frames, get_fitness(algo.best_individual), quantiles_values...)
        )
    end

    data_frame = DataFrames.DataFrame(
        list_with_results,
        [:generation, :total_evaluations, :total_frames, :best_fitness, percentiles_names...]
    )
    return data_frame
end

end