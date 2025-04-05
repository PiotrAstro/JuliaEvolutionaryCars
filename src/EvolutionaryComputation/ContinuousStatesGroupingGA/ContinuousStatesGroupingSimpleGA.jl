module ContinuousStatesGroupingSimpleGA

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

mutable struct Individual
    genes::Matrix{Float32}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    fihc_dict::Dict
    cross_dict::Dict
    _tree::StatesGrouping.TreeNode
    _trajectories::Vector{<:Environment.Trajectory}
    _trajectories_actual::Bool
    _fitness::Float64
    _fitness_actual::Bool
    _verbose::Bool
end

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, cross_dict::Dict, fihc_dict::Dict, initial_genes_mode::Symbol=:scale, verbose=false)
    genes = initial_genes(env_wrapper, initial_genes_mode)
    return Individual(
        genes,
        env_wrapper,
        fihc_dict,
        cross_dict,
        StatesGrouping.TreeNode(nothing, nothing, 1:EnvironmentWrapper.get_groups_number(env_wrapper)),
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
        ind.fihc_dict,
        ind.cross_dict,
        ind._tree,
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
        individual._trajectories, individual._tree = EnvironmentWrapper.create_time_distance_tree(individual.env_wrapper, individual.genes)
        individual._trajectories_actual = true
        individual._fitness_actual = true
        individual._fitness = sum([tra.rewards_sum for tra in individual._trajectories])
    end

    return individual._fitness
end

function crossover!(ind::Individual, other_individuals::Vector{Individual}) :: Int
    original_genes = Base.copy(ind.genes)

    other_n = length(other_individuals)
    genes_comb = get_genes_combinations(ind, ind.cross_dict[:genes_combinations])
    strategy = ind.cross_dict[:strategy]  # :one_rand or :one_tournament or :rand or :all_seq or :all_comb or rand_comb
    cross_strategy = ind.cross_dict[:cross_strategy]
    cross_prob = ind.cross_dict[:cross_prob]
    cross_prob_mode = ind.cross_dict[:cross_prob_mode]  # :all or :per_gene
    cross_f_value = ind.cross_dict[:f_value]
    evals = 0

    if strategy == :none
        return 0
    elseif strategy == :yes
        for nodes_level in genes_comb
            for node in Random.shuffle(nodes_level)
                accept_if_better!(ind, cross_strategy, cross_f_value, cross_prob, cross_prob_mode, other_individuals, node)
                evals += 1
            end
        end
    else
        throw(ArgumentError("Unknown strategy: $strategy"))
    end

    if original_genes != ind.genes
        ind._trajectories_actual = false
    end

    return evals
end

function copy_genes!(ind_to::Individual, ind_from::Individual)
    ind_to.genes = EnvironmentWrapper.translate(ind_from.env_wrapper, ind_from.genes, ind_to.env_wrapper)
    ind_to._fitness_actual = false
    ind_to._trajectories_actual = false
end

function accept_if_better!(ind::Individual, cross_strategy::Symbol, cross_f_value::Float64, cross_prob::Float64, cross_prob_mode::Symbol, others::Vector{Individual}, genes_changed::Vector{Float32})::Bool
    old_genes = Base.copy(ind.genes)
    old_fitness = get_fitness!(ind)

    genes_mask = get_genes_mask(ind, cross_prob, genes_changed, cross_prob_mode)
    base, other_1, other_2 = get_inidividuals_DE(ind, others, cross_strategy)
    new_genes = generate_new_genes_DE(ind, base, other_1, other_2, cross_f_value, genes_mask)
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

function get_genes_mask(ind::Individual, cross_prob::Float64, genes_changed::Vector{Float32}, mode::Symbol)::Matrix{Float32}
    genes_chenged_copy = ones(Float32, size(ind.genes))
    for (gene_col, changed_val) in zip(eachcol(genes_chenged_copy), genes_changed)
        gene_col .*= changed_val
    end

    if mode == :all
        for gene_id in eachindex(genes_chenged_copy)
            if rand() > cross_prob
                genes_chenged_copy[gene_id] = 0.0
            end
        end
    elseif mode == :column
        for gene_col in eachcol(genes_chenged_copy)
            if rand() > cross_prob
                gene_col .= 0.0
            end
        end
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
    return genes_chenged_copy
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
    self_factor, other_factor = ind.cross_dict[:self_vs_other]

    self_new_genes = Base.copy(ind.genes)
    self_new_genes .*= self_factor
    
    other_new_genes_base = EnvironmentWrapper.translate(other_base.env_wrapper, other_base.genes, ind.env_wrapper)
    other_1_new_genes = EnvironmentWrapper.translate(other_1.env_wrapper, other_1.genes, ind.env_wrapper)
    other_2_new_genes = EnvironmentWrapper.translate(other_2.env_wrapper, other_2.genes, ind.env_wrapper)
    other_new_genes = other_new_genes_base .+ f_value .* (other_1_new_genes .- other_2_new_genes)
    other_new_genes .*= other_factor

    new_genes = self_new_genes .+ other_new_genes

    final_new_genes = Base.copy(ind.genes)
    final_new_genes .*= 1.0f0 .- genes_changed
    final_new_genes .+= genes_changed .* new_genes
    final_new_genes = norm_genes(final_new_genes, ind.cross_dict[:norm_mode])
    return final_new_genes
end

function get_genes_combinations(ind::Individual, mode::Symbol)::Vector{Vector{Vector{Float32}}}
    genes_n = EnvironmentWrapper.get_groups_number(ind.env_wrapper)

    if mode == :all
        return [[ones(Float32, genes_n)]]
    elseif mode == :flat
        arr = [[zeros(Float32, genes_n) for _ in 1:genes_n]]
        for i in 1:genes_n
            arr[1][i][i] = 1.0f0
        end
        return arr
    elseif mode == :tree_up
        tree_up = get_genes_combinations(ind, :tree_down)
        return reverse(tree_up)
    elseif mode == :tree_down
        levels = Vector{Vector{Vector{Float32}}}()
        levels_nodes = Vector{Vector{StatesGrouping.TreeNode}}()
        push!(levels_nodes, [ind._tree.left, ind._tree.right])
        while true
            last_level = levels_nodes[end]
            new_level = Vector{StatesGrouping.TreeNode}()
            for node in last_level
                if !StatesGrouping.is_leaf(node)
                    push!(new_level, node.left)
                    push!(new_level, node.right)
                end
            end
            if isempty(new_level)
                break
            end
            push!(levels_nodes, new_level)
        end

        for level in levels_nodes
            level_distances = Vector{Vector{Float32}}()
            for node in level
                this_node_membership = zeros(Float32, genes_n)
                this_node_membership[node.elements] .= one(Float32)
                push!(level_distances, this_node_membership)
            end
            push!(levels, level_distances)
        end

        return levels
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end



function FIHC!(ind::Individual)::Int
    old_genes = Base.copy(ind.genes)
    eval_num = FIHC_test!(ind; ind.fihc_dict...)
    if ind.genes != old_genes
        ind._trajectories_actual = false
    end
    return eval_num
end

# --------------------------------------------------------------------------------------------------
# algorithm itself

mutable struct ContinuousStatesGroupingSimpleGA_Algorithm <: AbstractOptimizerModule.AbstractOptimizer
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    population::Vector{Individual}
    best_individual::Individual
    current_env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    run_statistics::Environment.RunStatistics
    fihc::Dict
    cross::Dict
    initial_genes_mode::Symbol
    verbose::Bool
    new_individual_each_n_epochs::Int
    new_individual_genes::Symbol
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
    individuals_n::Int,
    fihc::Dict,
    cross::Dict,
    initial_genes_mode::Symbol,
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

    individuals = [Individual(env_wrapper_struct, cross, fihc, initial_genes_mode) for _ in 1:individuals_n]
    # best_individual = individuals[1]
    Threads.@threads for ind in individuals
    # for ind in individuals
        ind.env_wrapper = EnvironmentWrapper.copy(env_wrapper_struct)
        EnvironmentWrapper.random_reinitialize_exemplars!(ind.env_wrapper)
        get_fitness!(ind)
        get_trajectories!(ind)
    end
    best_individual = individuals[argmax([get_fitness!(ind) for ind in individuals])]

    return ContinuousStatesGroupingSimpleGA_Algorithm(
        visualization_env,
        visualization_kwargs,
        individuals,
        best_individual,
        env_wrapper_struct,
        run_statistics,
        fihc,
        cross,
        initial_genes_mode,
        false,
        new_individual_each_n_epochs,
        new_individual_genes
    )
end

function AbstractOptimizerModule.run!(csgs::ContinuousStatesGroupingSimpleGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0) :: DataFrames.DataFrame
    # --------------------------------------------------
    # Test!!!
    # return run_test(csgs; max_generations=max_generations, max_evaluations=max_evaluations, log=log, fihc_settings=csgs.fihc)
    # --------------------------------------------------
    # Real implementation

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

        # just to get names
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

# import BenchmarkTools
# function get_trajectories_tmp(ind2)
#     ind2._trajectories_actual = false
#     get_trajectories!(ind2)
# end
# function fihc_tmp(ind2)
#     FIHC!(individual_copy(ind2))
# end
# function crossover_tmp(ind2, other2)
#     crossover!(individual_copy(ind2), other2)
# end

function create_new_env_wrap_and_individual(csgs::ContinuousStatesGroupingSimpleGA_Algorithm, individuals_copy_for_crossover::Vector{Individual})::Tuple{EnvironmentWrapper.EnvironmentWrapperStruct, Individual}
    new_env_wrapper, _ = EnvironmentWrapper.create_new_based_on(
        csgs.current_env_wrapper,
        [
            (1.0, get_flattened_trajectories(csgs.population)),
        ]
    )
    new_individual = Individual(new_env_wrapper, csgs.cross, csgs.fihc, csgs.initial_genes_mode, csgs.verbose)

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

function run_one_individual_generation!(ind::Individual, other::Vector{Individual})
    # println("\n\n\nTrajs:")
    # display(BenchmarkTools.@benchmark get_trajectories_tmp($ind))
    # println("\n\nFIHC:")
    # display(BenchmarkTools.@benchmark fihc_tmp($ind))
    # println("\n\nCrossover:")
    # display(BenchmarkTools.@benchmark crossover_tmp($ind, $other))
    # throw("dsdsvdsfvfdbjkfd")

    get_trajectories!(ind)
    evaluations = crossover!(ind, other)
    evaluations += FIHC!(ind)
    evaluations += ifelse(ind._trajectories_actual, 0, 1)
    get_trajectories!(ind)
end





# --------------------------------------------------------------------------------------------------
# tests

function run_test(csgs::ContinuousStatesGroupingSimpleGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, fihc_settings::Dict)
    new_ind = csgs.best_individual
    total_eval = 0

    # combinations_flat = get_genes_combination(new_ind.env_wrapper, :flat)
    # display(combinations_flat)
    # combinations_hier = get_genes_combination(new_ind.env_wrapper, :hier)
    # for level in combinations_hier
    #     display(level)
    # end
    # rand_matrix = generate_random_matrix(10, 5, 0.1, :randn)
    # norm_sum = norm_genes(rand_matrix, :d_sum)
    # display(norm_sum)
    # norm_min0 = norm_genes(rand_matrix, :min_0)
    # display(norm_min0)
    # throw("dsdsvdsfvfdbjkfd")
    if log
        Logging.@info("running test")
    end
    list_with_results = Vector{Tuple{Int, Int, Float64}}()
    push!(list_with_results, (0, total_eval, get_fitness_test!(new_ind)))
    for generation in 1:max_generations
        total_eval += FIHC_test!(new_ind; fihc_settings...)
        fitness_new = get_fitness_test!(new_ind)
        if log
            Logging.@info("Generation: $generation, total_eval: $total_eval, fitness: $fitness_new\n")
        end
        push!(list_with_results, (generation, total_eval, fitness_new))
        if total_eval >= max_evaluations
            break
        end
    end
    return DataFrames.DataFrame(
        list_with_results,
        [:generation, :total_evaluations, :best_fitness]
    )
end

function get_fitness_test!(ind::Individual) :: Float64
    # return 0.0  # Test!
    # if ind._fitness < 480.0
    #     get_fitness!(ind)
    # else
    #     return ind._fitness
    # end
    return get_fitness!(ind)
end

function FIHC_test!(ind::Individual;
    fihc_mode::Symbol,
    levels_mode::Symbol,
    norm_mode::Symbol,
    random_matrix_mode::Symbol,
    factor::Float64,
    hier_factor::Float64,
    local_fuzzy::Symbol
) :: Int

    if levels_mode == :distance
        levels = ind.env_wrapper._struct_memory._distance_membership_levels
    elseif levels_mode == :time_up
        levels = get_genes_combinations(ind, :tree_up)
    elseif levels_mode == :time_down
        levels = get_genes_combinations(ind, :tree_down)
    else
        throw(ArgumentError("Unknown mode: $levels_mode"))
    end
    # It doesnt work anyway, thus I do not want to keep updating it
    # if fihc_mode == :matrix_rand
    #     evals_num_now = EnvironmentWrapper.get_groups_number(ind.env_wrapper)  # I want to run few more so that I do not hav too many entries in logs
    #     for _ in evals_num_now
    #         old_fitness = get_fitness_test!(ind)
    #         old_genes = Base.copy(ind.genes)
    #         new_genes = ind.genes .+ generate_random_matrix(size(ind.genes, 1), size(ind.genes, 2), factor, random_matrix_mode)
    #         ind._fitness_actual = false
    #         ind.genes = norm_genes(new_genes, norm_mode)
            
    #         if get_fitness_test!(ind) < old_fitness
    #             ind.genes = old_genes
    #             ind._fitness = old_fitness
    #         end
    #     end
    #     return evals_num_now
    # elseif fihc_mode == :per_gene_rand
    evals = 0
    if fihc_mode == :none
        # will return 0 down
    elseif fihc_mode == :per_gene_rand
        evals = 0
        for nodes_level in levels
            for node in Random.shuffle(nodes_level)
                old_fitness = get_fitness_test!(ind)
                old_genes = Base.copy(ind.genes)
                random_values = generate_random_matrix(size(ind.genes, 1), size(ind.genes, 2), factor, random_matrix_mode)
                for (rand_col, membership) in zip(eachcol(random_values), node)
                    rand_col .*= membership
                end
                ind._fitness_actual = false
                ind.genes = norm_genes(old_genes .+ random_values, norm_mode)

                evals += 1
                
                if get_fitness_test!(ind) < old_fitness
                    ind.genes = old_genes
                    ind._fitness = old_fitness
                end
            end
        end
    
    elseif fihc_mode == :hier_decrease
        factor_tmp = factor
        for nodes_level in levels
            for node in Random.shuffle(nodes_level)
                old_fitness = get_fitness_test!(ind)
                old_genes = Base.copy(ind.genes)
                random_values = generate_random_matrix(size(ind.genes, 1), size(ind.genes, 2), factor_tmp, random_matrix_mode)
                for (rand_col, membership) in zip(eachcol(random_values), node)
                    rand_col .*= membership
                end
                ind._fitness_actual = false
                ind.genes = norm_genes(old_genes .+ random_values, norm_mode)

                evals += 1
                
                if get_fitness_test!(ind) < old_fitness
                    ind.genes = old_genes
                    ind._fitness = old_fitness
                end
            end
            factor_tmp *= hier_factor
        end

    elseif fihc_mode == :hier_increase
        genes_combinations = levels
        factors = Vector{Float64}(undef, length(genes_combinations))
        factors[1] = factor
        for i in 2:length(factors)
            factors[i] = factors[i - 1] * hier_factor
        end
        factors = reverse(factors)
        for (factor_tmp, nodes_level) in zip(factors, genes_combinations)
            for node in Random.shuffle(nodes_level)
                old_fitness = get_fitness_test!(ind)
                old_genes = Base.copy(ind.genes)
                random_values = generate_random_matrix(size(ind.genes, 1), size(ind.genes, 2), factor_tmp, random_matrix_mode)
                for (rand_col, membership) in zip(eachcol(random_values), node)
                    rand_col .*= membership
                end
                ind._fitness_actual = false
                ind.genes = norm_genes(old_genes .+ random_values, norm_mode)

                evals += 1
                
                if get_fitness_test!(ind) < old_fitness
                    ind.genes = old_genes
                    ind._fitness = old_fitness
                end
            end
        end

    # elseif fihc_mode == :disc_fihc
    #     for col in eachcol(ind.genes)
    #         argmax_col = argmax(col)
    #         col .= 0.0
    #         col[argmax_col] = 1.0
    #     end

    #     actions_n = EnvironmentWrapper.get_action_size(ind.env_wrapper)
    #     for nodes_level in ind.env_wrapper._struct_memory._distance_membership_levels
    #         for node in Random.shuffle(nodes_level)
    #             for action in 1:actions_n
    #                 old_fitness = get_fitness_test!(ind)
    #                 old_genes = Base.copy(ind.genes)
    #                 new_genes = Base.copy(ind.genes)
    #                 new_genes[:, node] .= 0.0
    #                 new_genes[action, node] .= 1.0
    #                 ind._fitness_actual = false
    #                 ind.genes = new_genes

    #                 evals += 1
                    
    #                 if get_fitness_test!(ind) < old_fitness
    #                     ind.genes = old_genes
    #                     ind._fitness = old_fitness
    #                 end
    #             end
    #         end
    #     end

    # elseif fihc_mode == :fihc_cont
    #     actions_n = EnvironmentWrapper.get_action_size(ind.env_wrapper)
    #     for nodes_level in ind.env_wrapper._struct_memory._distance_membership_levels
    #         for node in Random.shuffle(nodes_level)
    #             for action in 1:actions_n
    #                 old_fitness = get_fitness_test!(ind)
    #                 old_genes = Base.copy(ind.genes)
    #                 genes_changes = zeros(Float32, size(ind.genes, 1), length(node))
    #                 genes_changes[action, :] .+= factor
    #                 ind._fitness_actual = false
    #                 ind.genes = norm_genes(old_genes, genes_changes, node, norm_mode)

    #                 evals += 1
                    
    #                 if get_fitness_test!(ind) < old_fitness
    #                     ind.genes = old_genes
    #                     ind._fitness = old_fitness
    #                 end
    #             end
    #         end
    #     end

    else
        throw(ArgumentError("Unknown mode: $fihc_mode"))
    end

    evals += local_fuzzy_FIHC!(ind, local_fuzzy)
    return evals
end

function local_fuzzy_FIHC!(ind::Individual, mode::Symbol)
    if mode == :none
        return 0
    elseif mode == :global
        new_genes = Base.copy(ind.genes)
        for gene_id in 1:EnvironmentWrapper.get_groups_number(ind.env_wrapper)
            new_genes[:, gene_id] .= local_get_fuzzier_gene(ind, gene_id)
        end
        old_genes = ind.genes
        old_fitness = get_fitness_test!(ind)
        ind.genes = new_genes
        ind._fitness_actual = false
        if get_fitness_test!(ind) < old_fitness
            ind.genes = old_genes
            ind._fitness = old_fitness
        end
        return 1
    elseif mode == :per_gene
        evals = 0
        for gene_id in Random.randperm(EnvironmentWrapper.get_groups_number(ind.env_wrapper))
            new_genes = Base.copy(ind.genes)
            new_genes[:, gene_id] = local_get_fuzzier_gene(ind, gene_id)
            old_genes = ind.genes
            old_fitness = get_fitness_test!(ind)
            ind.genes = new_genes
            ind._fitness_actual = false
            evals += 1
            if get_fitness_test!(ind) < old_fitness
                ind.genes = old_genes
                ind._fitness = old_fitness
            end
        end
        return evals
    elseif mode == :always
        new_genes = Base.copy(ind.genes)
        for gene_id in 1:EnvironmentWrapper.get_groups_number(ind.env_wrapper)
            new_genes[:, gene_id] .= local_get_fuzzier_gene(ind, gene_id)
        end
        ind.genes = new_genes
        ind._fitness_actual = false
        get_fitness_test!(ind)
        return 1
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end

function local_get_fuzzier_gene(ind::Individual, gene_id)
    genes_n = EnvironmentWrapper.get_groups_number(ind.env_wrapper)
    this_exemplar = ind.env_wrapper._encoded_exemplars[:, gene_id:gene_id]
    other_exemplars = ind.env_wrapper._encoded_exemplars[:, (1:genes_n) .!= gene_id]
    old_genes_not_this_one = ind.genes[:, (1:genes_n) .!= gene_id]
    memberships = StatesGrouping.get_membership(this_exemplar, other_exemplars, ind.env_wrapper._distance_metric, ind.env_wrapper._m_value)
    new_gene_value = old_genes_not_this_one * memberships
    new_gene_value ./= sum(new_gene_value)
    return new_gene_value
end

function initial_genes(env_wrap::EnvironmentWrapper.EnvironmentWrapperStruct, mode::Symbol) :: Matrix{Float32}
    new_genes = randn(Float32, EnvironmentWrapper.get_action_size(env_wrap), EnvironmentWrapper.get_groups_number(env_wrap))
    norm_genes(new_genes, mode)

    return new_genes
end

function norm_genes(genes_origianal::Matrix{Float32}, mode::Symbol) :: Matrix{Float32}
    new_genes = Base.copy(genes_origianal)
    if mode == :d_sum
        EnvironmentWrapper.normalize_genes!(new_genes)
    elseif mode == :min_0
        EnvironmentWrapper.normalize_genes_min_0!(new_genes)
    elseif mode == :none
        # pass
    elseif mode == :std
        znorm!(new_genes)
    # elseif mode == :around_0
    #     # we transform genes to mean 0 std 0, than add randn and then normalize  -= minimum  and  / sum
    #     genes_changed = genes_origianal[:, changed_genes]
    #     znorm!(genes_changed)
    #     genes_changed += genes_new
    #     znorm!(genes_changed)
    #     EnvironmentWrapper.normalize_genes_min_0!(genes_changed)
    #     new_genes[:, changed_genes] = genes_changed
    # elseif mode == :softmax_norm
    #     # we get original ifferences between values from previous softmax, than add randn and then normalize with softmax
    #     genes_changed = genes_origianal[:, changed_genes]
    #     softmax_inv!(genes_changed)
    #     znorm!(genes_changed)
    #     genes_changed += genes_new
    #     znorm!(genes_changed)
    #     softmax!(genes_changed)
    #     new_genes[:, changed_genes] = genes_changed
    # elseif mode == :softmax
    #     genes_changed = genes_origianal[:, changed_genes]
    #     softmax_inv!(genes_changed)
    #     genes_changed += genes_new
    #     softmax!(genes_changed)
    #     new_genes[:, changed_genes] = genes_changed
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

function generate_random_matrix(latent_space::Int, n_clusters::Int, factor::Float64, mode::Symbol) :: Matrix{Float32}
    if mode == :rand_different
        return rand(Float32, latent_space, n_clusters) .* factor
    elseif mode == :rand_n_different
        return randn(Float32, latent_space, n_clusters) .* factor
    elseif mode == :rand_same
        vector = rand(Float32, latent_space) .* factor
        return repeat(vector, 1, n_clusters)
    elseif mode == :rand_n_same
        vector = randn(Float32, latent_space) .* factor
        return repeat(vector, 1, n_clusters)
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end


end # module ContinuousStatesGroupingP3