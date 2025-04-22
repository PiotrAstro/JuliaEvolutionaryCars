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

include("EnvironmentWrapper.jl")
import .EnvironmentWrapper

# --------------------------------------------------------------------------------------------------
# individuals

@kwdef struct IndividualConfig
    initial_genes_mode::Symbol=:scale
    norm_genes::Symbol=:std
    levels_mode::Symbol = :time_markov
    levels_hclust::Symbol = :average
    levels_construct_mode::Symbol = :equal_up
    base_mode::Symbol=:best
    mask_mode::Symbol=:per_gene
    cross_n_times::Int=1
    cross_f::Float64 = 0.8
    cross_prob::Float64 = 1.0
end

mutable struct Individual
    genes::Matrix{Float32}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    config::IndividualConfig
    _levels::Vector{Vector{Vector{Int}}}
    _trajectories::Vector{<:Environment.Trajectory}
    _trajectories_actual::Bool
    _fitness::Float64
    _fitness_actual::Bool
    _verbose::Bool
end

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, individual_config::Dict, verbose::Bool=false)::Individual
    config = IndividualConfig(;individual_config...)
    genes = initial_genes(env_wrapper, config.initial_genes_mode)
    return Individual(
        genes,
        env_wrapper,
        config,
        Vector{Vector{Vector{Int}}}(),
        Vector{Environment.Trajectory}(undef, 0),
        false,
        -Inf64,
        false,
        verbose
    )
end

function run_one_individual_generation!(ind::Individual, other::Vector{Individual})
    # println("\n\n\nTrajs:")
    # display(BenchmarkTools.@benchmark get_trajectories_tmp($ind))
    # println("\n\nFIHC:")
    # display(BenchmarkTools.@benchmark fihc_tmp($ind))
    # println("\n\nCrossover:")
    # display(BenchmarkTools.@benchmark crossover_tmp($ind, $other))
    # throw("dsdsvdsfvfdbjkfd")

    get_trajectories!(ind)
    for _ in 1:ind.config.cross_n_times
        crossover!(ind, other)
    end
    get_trajectories!(ind)
end

function individual_copy(ind::Individual)
    Individual(
        Base.copy(ind.genes),
        ind.env_wrapper,
        ind.config,
        ind._levels,
        ind._trajectories,
        ind._trajectories_actual,
        ind._fitness,
        ind._fitness_actual,
        ind._verbose
    )
end

function get_flattened_trajectories(individuals::Vector{Individual})::Vector{<:Environment.Trajectory}
    return reduce(vcat, [individual._trajectories for individual in individuals])
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
        individual._levels = _get_levels(individual)
        individual._trajectories_actual = true
        individual._fitness_actual = true
        individual._fitness = sum([tra.rewards_sum for tra in individual._trajectories])
    end

    return individual._fitness
end

function _get_levels(individual::Individual)
    mode = individual.config.levels_mode
    exemplars_n = size(individual.genes, 2)

    if mode == :all
        return [[ones(Int, exemplars_n)]]
    elseif mode == :flat
        arr = [[zeros(Int, exemplars_n) for _ in 1:exemplars_n]]
        for i in 1:exemplars_n
            arr[1][i][i] = 1
        end
        return arr
    elseif mode == :latent
        return EnvironmentWrapper.get_levels_latent(
            individual.env_wrapper,
            individual.config.levels_construct_mode,
            individual.config.levels_hclust
        )
    elseif mode == :time_markov
        return EnvironmentWrapper.get_levels_time(
            individual.env_wrapper,
            individual.genes,
            :markov,
            individual.config.levels_construct_mode,
            individual.config.levels_hclust;
            trajectories=individual._trajectories
        )
    elseif mode == :time_mine
        return EnvironmentWrapper.get_levels_time(
            individual.env_wrapper,
            individual.genes,
            :mine,
            individual.config.levels_construct_mode,
            individual.config.levels_hclust;
            individual.trajectories
        )
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end

function crossover!(ind::Individual, other_individuals::Vector{Individual})
    for nodes_level in ind._levels
        for node in Random.shuffle(nodes_level)
            accept_if_better!(ind, node, other_individuals)
        end
    end
end

function copy_genes!(ind_to::Individual, ind_from::Individual)
    ind_to.genes = EnvironmentWrapper.translate(ind_from.env_wrapper, ind_from.genes, ind_to.env_wrapper)
    ind_to._fitness_actual = false
    ind_to._trajectories_actual = false
end

function accept_if_better!(ind::Individual, genes_changed::Vector{Int}, others::Vector{Individual})::Bool
    old_genes = Base.copy(ind.genes)
    old_fitness = get_fitness!(ind)

    genes_mask = get_genes_mask(ind, ind.config.cross_prob, genes_changed, ind.config.mask_mode)
    base, other_1, other_2 = get_inidividuals_DE(ind, others, ind.config.base_mode)
    new_genes = generate_new_genes_DE(ind, base, other_1, other_2, ind.config.cross_f, genes_mask)
    ind.genes = new_genes
    ind._fitness_actual = false
    new_fitness = get_fitness!(ind)

    if new_fitness < old_fitness
        ind.genes = old_genes
        ind._fitness = old_fitness
        return false
    else
        ind._trajectories_actual = false
        return true
    end
end

function get_genes_mask(ind::Individual, cross_prob::Float64, genes_changed::Vector{Int}, mask_mode::Symbol)::Matrix{Float32}
    genes_changed_copy = zeros(Float32, size(ind.genes))
    for gene_id in eachindex(genes_changed)
        genes_changed_copy[:, gene_id] .= 1.0f0
    end

    if mask_mode == :per_value
        for gene_id in eachindex(genes_changed_copy)
            if rand() > cross_prob
                genes_changed_copy[gene_id] = 0.0
            end
        end
    elseif mask_mode == :per_gene
        for gene_col in eachcol(genes_changed_copy)
            if rand() > cross_prob
                gene_col .= 0.0
            end
        end
    else
        throw(ArgumentError("Unknown mode: $mask_mode"))
    end
    return genes_changed_copy
end

function get_inidividuals_DE(ind::Individual, others::Vector{Individual}, mode::Symbol)::Tuple{Individual, Individual, Individual}
    other_1 = others[rand(1:length(others))]
    while other_1 === ind
        other_1 = others[rand(1:length(others))]
    end

    other_2 = others[rand(1:length(others))]
    while other_2 === ind || other_2 === other_1
        other_2 = others[rand(1:length(others))]
    end

    if mode == :rand
        other_base = others[rand(1:length(others))]
        while other_base === ind || other_base === other_1 || other_base === other_2
            other_base = others[rand(1:length(others))]
        end
    elseif mode == :best
        other_base = others[argmax(get_fitness!(other) for other in others)]
    elseif mode == :self
        other_base = ind
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end

    return other_base, other_1, other_2
end

function generate_new_genes_DE(ind::Individual, other_base::Individual, other_1::Individual, other_2::Individual, f_value::Float64, genes_changed::Matrix{Float32})::Matrix{Float32}    
    other_new_genes_base = EnvironmentWrapper.translate(other_base.env_wrapper, other_base.genes, ind.env_wrapper)
    other_1_new_genes = EnvironmentWrapper.translate(other_1.env_wrapper, other_1.genes, ind.env_wrapper)
    other_2_new_genes = EnvironmentWrapper.translate(other_2.env_wrapper, other_2.genes, ind.env_wrapper)
    other_new_genes = other_new_genes_base .+ f_value .* (other_1_new_genes .- other_2_new_genes)

    final_new_genes = Base.copy(ind.genes)
    final_new_genes .*= 1.0f0 .- genes_changed
    final_new_genes .+= genes_changed .* other_new_genes
    final_new_genes = norm_genes(final_new_genes, ind.config.norm_genes)
    return final_new_genes
end

# --------------------------------------------------------------------------------------------------
# algorithm itself

mutable struct ContinuousStatesGroupingDE_Algorithm <: AbstractOptimizerModule.AbstractOptimizer
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    population::Vector{Individual}
    best_individual::Individual
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
    

    env_wrapper_struct, _ = EnvironmentWrapper.EnvironmentWrapperStruct(
        environments,
        run_statistics
        ;
        env_wrapper...
    )

    individuals = [Individual(env_wrapper_struct, individual_config) for _ in 1:individuals_n]
    # best_individual = individuals[1]
    # Threads.@threads for ind in individuals
    for ind in individuals
        ind.env_wrapper = EnvironmentWrapper.copy(env_wrapper_struct)
        EnvironmentWrapper.random_reinitialize_exemplars!(ind.env_wrapper)
        get_trajectories!(ind)
    end
    best_individual = individuals[argmax([get_fitness!(ind) for ind in individuals])]

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

        individuals_copy_for_crossover = [individual_copy(ind) for ind in csgs.population]

        new_env_wrap = csgs.current_env_wrapper
        new_individual = csgs.population[1]

        should_add_individual = generation % csgs.new_individual_each_n_epochs == 0
        start_from_n = ifelse(should_add_individual, 0, 1)
        Threads.@threads :dynamic for i in start_from_n:length(csgs.population)
            if i == 0
                new_env_wrap, new_individual = create_new_env_wrap_and_individual(csgs, individuals_copy_for_crossover)
            else
                run_one_individual_generation!(csgs.population[i], individuals_copy_for_crossover)
            end
        end
        best_ind_arg = argmax(get_fitness!.(csgs.population))
        csgs.best_individual = individual_copy(csgs.population[best_ind_arg])

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

        best_fitness = get_fitness!(csgs.best_individual)
        fitnesses = get_fitness!.(csgs.population)
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

function create_new_env_wrap_and_individual(csgs::ContinuousStatesGroupingDE_Algorithm, individuals_copy_for_crossover::Vector{Individual})::Tuple{EnvironmentWrapper.EnvironmentWrapperStruct, Individual}
    new_env_wrapper, _ = EnvironmentWrapper.create_new_based_on(
        csgs.current_env_wrapper,
        [
            (1.0, get_flattened_trajectories(csgs.population)),
        ]
    )
    new_individual = Individual(new_env_wrapper, csgs.individual_config, csgs.verbose)

    if csgs.new_individual_genes == :best
        copy_genes!(new_individual, csgs.best_individual)
    elseif csgs.new_individual_genes == :rand
        # pass
    else
        throw(ArgumentError("Unknown new_individual_genes: $(csgs.new_individual_genes)"))
    end

    get_trajectories!(new_individual)
    return new_env_wrapper, new_individual
end

# --------------------------------------------------------------------------------------------------
# math

function initial_genes(env_wrap::EnvironmentWrapper.EnvironmentWrapperStruct, mode::Symbol) :: Matrix{Float32}
    new_genes = randn(Float32, EnvironmentWrapper.get_action_size(env_wrap), EnvironmentWrapper.get_groups_number(env_wrap))
    norm_genes(new_genes, mode)

    return new_genes
end

function norm_genes(genes_origianal::Matrix{Float32}, mode::Symbol) :: Matrix{Float32}
    new_genes = Base.copy(genes_origianal)
    if mode == :d_sum
        dsum_norm!(new_genes)
    elseif mode == :min_0
        min_0_norm!(new_genes)
    elseif mode == :none
        # pass
    elseif mode == :std
        znorm!(new_genes)
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
    return new_genes
end

function znorm!(genes::Matrix{Float32})
    for col in eachcol(genes)
        col .-= Statistics.mean(col)
        col ./= (Statistics.std(col) + eps(Float32))
    end
end

"""
It normalizes genes by making it non negative and sum to 0
opposed to normalize_genes_min_0! if input is e.g. 0.2 0.4 0.4 it will stay the same
"""
function min_0_norm!(genes::Matrix{Float32})
    for col in eachcol(genes)
        min_value = minimum(col)
        col .-= min_value
        col ./= sum(col)
    end
end

"""
It normalizes genes by subtracting smallest value from each col and then subtracting by sum
if input is e.g. 0.2 0.4 0.4 it will be normalized to 0 0.5 0.5
"""
function dsum_norm!(genes::Matrix{Float32})
    for col in eachcol(genes)
        min_value = minimum(col)
        if min_value < 0
            col .+= abs(min_value)
        end
        col ./= sum(col)
    end
end

function softmax!(genes::Matrix{Float32})
    for col in eachcol(genes)
        col .-= minimum(col)
        col .= exp.(col)
        col ./= (sum(col) + eps(Float32))
    end
end

function softmax_inv!(genes::Matrix{Float32})
    for col in eachcol(genes)
        col .= log.(col .+ eps(Float32))
    end
end

end # module ContinuousStatesGroupingDE