module Individuals

import Statistics
import Random

import ..Environment
import ..NeuralNetwork
import ..StatesGrouping
import ..EnvironmentWrapper
import ..fitnesses_reduction

export Individual, IndividualConfig, run_one_individual_generation!, get_fitness!, get_trajectories!,
       crossover!, accept_if_better!, individual_copy, get_flattened_trajectories,
       get_genes_mask, generate_new_genes_DE, get_inidividuals_DE, copy_genes!, get_nn, FIHC_generation!, set_genes!

# --------------------------------------------------------------------------------------------------
# individuals

@kwdef struct IndividualConfig
    initial_genes_mode::Symbol=:none
    norm_genes::Symbol=:std
    levels_mode::Symbol = :time_markov
    levels_hclust::Symbol = :average
    levels_construct_mode::Symbol = :equal_up
    base_mode::Symbol=:best
    mask_mode::Symbol=:per_gene
    cross_n_times::Int=1
    cross_f::Float64 = 0.8
    cross_prob::Float64 = 1.0
    fitnesses_reduction::Symbol = :sum

    per_column_norm::Bool = true

    # FIHC
    fihc_f::Float64 = 0.1
    fihc_n_times::Int = 1
    fihc_same_per_gene::Bool = true
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
    genes = initial_genes(env_wrapper, config.initial_genes_mode, config.per_column_norm)
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

function set_genes!(ind::Individual, genes::Matrix{Float32})
    ind.genes = genes
    ind._fitness_actual = false
    ind._trajectories_actual = false
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
        individual._levels = _get_levels(individual)
        individual._trajectories_actual = true
        individual._fitness_actual = true
        reduction_mehod = individual.config.fitnesses_reduction
        individual._fitness = fitnesses_reduction(reduction_mehod, [tra.rewards_sum for tra in individual._trajectories])
    end

    return individual._fitness
end

function _get_levels(individual::Individual)::Vector{Vector{Vector{Int}}}
    mode = individual.config.levels_mode
    exemplars_n = size(individual.genes, 2)

    if mode == :all
        return [[collect(1:exemplars_n)]]
    elseif mode == :flat
        return [[[i] for i in 1:exemplars_n]]
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
            trajectories=individual._trajectories
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
    for gene in genes_changed
        genes_changed_copy[:, gene] .= 1.0f0
    end

    if mask_mode == :per_value
        for gene_id in eachindex(genes_changed_copy)
            if rand() > cross_prob
                genes_changed_copy[gene_id] = 0.0f0
            end
        end
    elseif mask_mode == :per_gene
        for gene_col in eachcol(genes_changed_copy)
            if rand() > cross_prob
                gene_col .= 0.0f0
            end
        end
    else
        throw(ArgumentError("Unknown mode: $mask_mode"))
    end
    return genes_changed_copy
end

function get_inidividuals_DE(ind::Individual, others::Vector{Individual}, mode::Symbol)::Tuple{Individual, Individual, Individual}
    perm = Random.randperm(length(others))
    other_1 = others[perm[1]]
    other_2 = others[perm[2]]

    if mode == :rand
        other_base = others[perm[3]]
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
    final_new_genes = norm_genes(final_new_genes, ind.config.norm_genes, ind.config.per_column_norm)
    return final_new_genes
end

# ---------------------------------------------------------------------------------------------------
# FIHC functions

"""
It performs defined number of FIHC iterations
It will make sure that the object generates trajectories
"""
function FIHC_generation!(ind::Individual)
    get_trajectories!(ind)
    for _ in 1:ind.config.fihc_n_times
        _FIHC!(ind)
    end
    get_trajectories!(ind)

    return true
end

function _FIHC!(ind::Individual)
    for nodes_level in ind._levels
        for node in Random.shuffle(nodes_level)
            old_genes = Base.copy(ind.genes)
            old_fitness = get_fitness!(ind)

            new_genes = Base.copy(ind.genes)
            if ind.config.fihc_same_per_gene
                random_genes = randn(Float32, size(new_genes, 1)) .* ind.config.fihc_f
                for gene in node
                    new_genes[:, gene] .+= random_genes
                end
            else
                for gene in node
                    new_genes[:, gene] .+= randn(Float32, size(new_genes, 1)) .* ind.config.fihc_f
                end
            end

            new_genes = norm_genes(new_genes, ind.config.norm_genes, ind.config.per_column_norm)
            ind.genes = new_genes
            ind._fitness_actual = false
            new_fitness = get_fitness!(ind)
            if new_fitness < old_fitness
                ind.genes = old_genes
                ind._fitness = old_fitness
            else
                ind._trajectories_actual = false
            end
        end
    end
end



# --------------------------------------------------------------------------------------------------
# math

function initial_genes(env_wrap::EnvironmentWrapper.EnvironmentWrapperStruct, mode::Symbol, per_column::Bool) :: Matrix{Float32}
    new_genes = randn(Float32, EnvironmentWrapper.get_action_size(env_wrap), EnvironmentWrapper.get_groups_number(env_wrap))
    norm_genes(new_genes, mode, per_column)

    return new_genes
end

function norm_genes(genes_origianal::Matrix{Float32}, mode::Symbol, per_column::Bool) :: Matrix{Float32}
    new_genes = Base.copy(genes_origianal)
    if !per_column
        new_genes = collect(new_genes')
    end

    if mode == :d_sum
        dsum_norm!(new_genes)
    elseif mode == :min_0
        min_0_norm!(new_genes)
    elseif mode == :none
        # pass
    elseif mode == :tanh
        tanh_norm!(new_genes)
    elseif mode == :std
        znorm!(new_genes)
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end

    if !per_column
        new_genes = collect(new_genes')
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

function tanh_norm!(genes::Matrix{Float32})
    genes .= tanh.(genes)
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



end