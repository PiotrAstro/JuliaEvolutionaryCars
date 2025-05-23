module ContinuousStatesGroupingP3

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
    _tree::Union{StatesGrouping.TreeNode, Nothing}
    _trajectories::Union{Vector{<:Environment.Trajectory}, Nothing}
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
        nothing,
        nothing,
        false,
        -Inf64,
        false,
        verbose
    )
end

"""
It removes memory that will no longer be used
"""
function clean!(ind::Individual)
    ind._trajectories = nothing
    ind._trajectories_actual = false
    ind._tree = nothing
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
    evals = 0

    if strategy == :none
        return 0
    elseif strategy == :one_rand
        other = other_individuals[rand(1:other_n)]
        for nodes_level in genes_comb
            for node in Random.shuffle(nodes_level)
                accept_if_better!(ind, other, node)
                evals += 1
            end
        end
    elseif strategy == :all_seq
        for other in Random.shuffle(other_individuals)
            for nodes_level in genes_comb
                for node in Random.shuffle(nodes_level)
                    accept_if_better!(ind, other, node)
                    evals += 1
                end
            end
        end
    elseif strategy == :all_comb
        for nodes_level in genes_comb
            for node in Random.shuffle(nodes_level)
                for other in Random.shuffle(other_individuals)
                    accept_if_better!(ind, other, node)
                    evals += 1
                end
            end
        end
    elseif strategy == :rand_comb
        for nodes_level in genes_comb
            for node in Random.shuffle(nodes_level)
                other = other_individuals[rand(1:other_n)]
                accept_if_better!(ind, other, node)
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

function accept_if_better!(ind::Individual, other::Individual, genes_changed::Vector{Float32})::Bool
    old_genes = Base.copy(ind.genes)
    old_fitness = get_fitness!(ind)
    new_genes = generate_new_genes(ind, other, genes_changed)
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

function generate_new_genes(ind::Individual, other::Individual, genes_changed::Vector{Float32})::Matrix{Float32}
    self_factor, other_factor = ind.cross_dict[:self_vs_other]

    self_new_genes = Base.copy(ind.genes)
    self_new_genes .*= self_factor
    
    other_new_genes = EnvironmentWrapper.translate(other.env_wrapper, other.genes, ind.env_wrapper)
    other_new_genes .*= other_factor

    new_genes = norm_genes(self_new_genes, other_new_genes, ind.cross_dict[:norm_mode])

    final_new_genes = Base.copy(ind.genes)
    for (final_col, new_col, changed_val) in zip(eachcol(final_new_genes), eachcol(new_genes), genes_changed)
        final_col .*= 1.0 - changed_val
        final_col .+= changed_val .* new_col
    end

    return final_new_genes
end


function copy_genes!(ind_to::Individual, ind_from::Individual)
    ind_to.genes = EnvironmentWrapper.translate(ind_from.env_wrapper, ind_from.genes, ind_to.env_wrapper)
    ind_to._fitness_actual = false
    ind_to._trajectories_actual = false
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

mutable struct ContinuousStatesGroupingP3_Algorithm <: AbstractOptimizerModule.AbstractOptimizer
    visualization_env::Environment.AbstractEnvironment
    visualization_kwargs::Dict{Symbol, Any}
    populations::Vector{Vector{Individual}}
    best_individual::Individual
    current_env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    current_states_sequence
    total_evaluations::Int
    fihc::Dict
    cross::Dict
    initial_genes_mode::Symbol
    verbose::Bool
end

function AbstractOptimizerModule.get_optimizer(::Val{:ContinuousStatesGroupingP3})
    return ContinuousStatesGroupingP3_Algorithm
end

function ContinuousStatesGroupingP3_Algorithm(;
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    env_wrapper::Dict{Symbol, <:Any},
    fihc::Dict,
    cross::Dict,
    initial_genes_mode::Symbol
)
    environment_type = Environment.get_environment(environment)
    environments = [(environment_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]
    visualization_env = (environment_type)(;environment_visualization_kwargs...)
    

    env_wrapper_struct, current_states_sequence = EnvironmentWrapper.EnvironmentWrapperStruct(
        environments;
        env_wrapper...
    )
    best_individual = Individual(env_wrapper_struct, cross, fihc, initial_genes_mode)
    FIHC!(best_individual)

    return ContinuousStatesGroupingP3_Algorithm(
        visualization_env,
        visualization_kwargs,
        [[best_individual]],
        best_individual,
        env_wrapper_struct,
        current_states_sequence,
        0,
        fihc,
        cross,
        initial_genes_mode,
        false
    )
end

function AbstractOptimizerModule.run!(csgs::ContinuousStatesGroupingP3_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0) :: DataFrames.DataFrame
    # --------------------------------------------------
    # Test!!!
    # return run_test(csgs; max_generations=max_generations, max_evaluations=max_evaluations, log=log, fihc_settings=csgs.fihc)
    # --------------------------------------------------
    # Real implementation

    EnvironmentWrapper.set_verbose!(csgs.current_env_wrapper, log)
    csgs.verbose = log
    csgs.populations[1][1]._verbose = log

    # (generation, total_evaluations, best_fitness, new_ind_fitness)
    list_with_results = Vector{Tuple{Int, Int, Float64, Float64}}()

    previous_individual = csgs.populations[1][1]

    for generation in 1:max_generations
        if csgs.total_evaluations >= max_evaluations
            break
        end

        Base.GC.gc(true)  # we want to reclaim all possible memory, true means we will do full GC
        start_time = time()

        new_individual = Individual(csgs.current_env_wrapper, csgs.cross, csgs.fihc, csgs.initial_genes_mode, csgs.verbose)
        copy_genes!(new_individual, csgs.best_individual)
        new_individual = run_new_individual!(new_individual, csgs)

        csgs.current_env_wrapper, csgs.current_states_sequence = EnvironmentWrapper.create_new_based_on(
            csgs.current_env_wrapper,
            [
                (0.9, [EnvironmentWrapper.Environment.Trajectory(csgs.current_states_sequence)]),
                (0.1, get_flattened_trajectories([new_individual]))
            ]
        )

        clean!(previous_individual)
        EnvironmentWrapper.clean!(previous_individual.env_wrapper)
        previous_individual = new_individual
        
        if get_fitness!(new_individual) > get_fitness!(csgs.best_individual)
            csgs.best_individual = individual_copy(new_individual)
        end

        end_time = time()

        # if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
        #     Ind.visualize(p3.best_individual, p3.visualization_env, p3.visualization_kwargs)
        # end

        best_fitness = get_fitness!(csgs.best_individual)
        new_ind_fitness = get_fitness!(new_individual)
        if log
            elapsed_time = end_time - start_time
            Logging.@info "\n\n\n\n\n\nGeneration $generation\nTotal evaluations: $(csgs.total_evaluations)\n" *
            Printf.@sprintf("elapsed_time: %.2f s\nbest_fitness: %.2f\nnew_fitness: %.2f\n\n\n", elapsed_time, best_fitness, new_ind_fitness)
        end
        
        push!(
            list_with_results,
            (generation, csgs.total_evaluations, best_fitness, new_ind_fitness)
        )
    end

    data_frame = DataFrames.DataFrame(
        list_with_results,
        [:generation, :total_evaluations, :best_fitness, :new_ind_fitness]
    )
    return data_frame
end

function run_new_individual!(new_individual::Individual, csgs::ContinuousStatesGroupingP3_Algorithm)::Individual
    csgs.total_evaluations += FIHC!(new_individual)
    add_to_next_level = true
    pop_id = 1
    while pop_id <= length(csgs.populations)
        should_mix = length(csgs.populations[pop_id]) > 0
        if add_to_next_level
            new_ind_copy = individual_copy(new_individual)
            clean!(new_ind_copy)
            push!(csgs.populations[pop_id], new_ind_copy)
            if csgs.verbose
                Logging.@info "adding new individual to $pop_id level\n"
            end
        end
        if should_mix
            previous_fitness = get_fitness!(new_individual)
            if csgs.verbose
                Logging.@info "mixing new individual with $pop_id level\n"
            end
            csgs.total_evaluations += run_one_individual_generation(new_individual, csgs.populations[pop_id])
            new_fitness = get_fitness!(new_individual)
            add_to_next_level = previous_fitness < new_fitness

            if add_to_next_level && pop_id == length(csgs.populations)
                push!(csgs.populations, Vector{Individual}())
            end
        end
        pop_id += 1
    end
    return new_individual
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

function run_one_individual_generation(ind::Individual, other::Vector{Individual})::Int
    # println("\n\n\nTrajs:")
    # display(BenchmarkTools.@benchmark get_trajectories_tmp($ind))
    # println("\n\nFIHC:")
    # display(BenchmarkTools.@benchmark fihc_tmp($ind))
    # println("\n\nCrossover:")
    # display(BenchmarkTools.@benchmark crossover_tmp($ind, $other))
    # throw("dsdsvdsfvfdbjkfd")

    get_trajectories!(ind)
    pre_fitness = get_fitness!(ind)

    evaluations = crossover!(ind, other)
    cross_fitness = get_fitness!(ind)

    evaluations += FIHC!(ind)
    fihc_fitness = get_fitness!(ind)

    evaluations += ifelse(ind._trajectories_actual, 0, 1)
    get_trajectories!(ind)

    if ind._verbose
        Logging.@info Printf.@sprintf("pre: %.2f  cross: %.2f  fihc: %.2f\n", pre_fitness, cross_fitness, fihc_fitness)
    end
    return evaluations
end





# --------------------------------------------------------------------------------------------------
# tests

function run_test(csgs::ContinuousStatesGroupingP3_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, fihc_settings::Dict)
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
    norm_mode::Symbol,
    random_matrix_mode::Symbol,
    factor::Float64,
    hier_factor::Float64,
    local_fuzzy::Symbol
) :: Int

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
    if fihc_mode == :per_gene_rand
        evals = 0
        for nodes_level in ind.env_wrapper._struct_memory._distance_membership_levels
            for node in Random.shuffle(nodes_level)
                old_fitness = get_fitness_test!(ind)
                old_genes = Base.copy(ind.genes)
                random_values = generate_random_matrix(size(ind.genes, 1), size(ind.genes, 2), factor, random_matrix_mode)
                for (rand_col, membership) in zip(eachcol(random_values), node)
                    rand_col .*= membership
                end
                ind._fitness_actual = false
                ind.genes = norm_genes(old_genes, random_values, norm_mode)

                evals += 1
                
                if get_fitness_test!(ind) < old_fitness
                    ind.genes = old_genes
                    ind._fitness = old_fitness
                end
            end
        end
    
    elseif fihc_mode == :hier_decrease
        factor_tmp = factor
        for nodes_level in ind.env_wrapper._struct_memory._distance_membership_levels
            for node in Random.shuffle(nodes_level)
                old_fitness = get_fitness_test!(ind)
                old_genes = Base.copy(ind.genes)
                random_values = generate_random_matrix(size(ind.genes, 1), size(ind.genes, 2), factor_tmp, random_matrix_mode)
                for (rand_col, membership) in zip(eachcol(random_values), node)
                    rand_col .*= membership
                end
                ind._fitness_actual = false
                ind.genes = norm_genes(old_genes, random_values, norm_mode)

                evals += 1
                
                if get_fitness_test!(ind) < old_fitness
                    ind.genes = old_genes
                    ind._fitness = old_fitness
                end
            end
            factor_tmp *= hier_factor
        end

    elseif fihc_mode == :hier_increase
        genes_combinations = ind.env_wrapper._struct_memory._distance_membership_levels
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
                ind.genes = norm_genes(old_genes, random_values, norm_mode)

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
    if mode == :scale
        EnvironmentWrapper.normalize_genes!(new_genes)
    elseif mode == :softmax
        softmax!(new_genes)
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end

    return new_genes
end

function norm_genes(genes_origianal::Matrix{Float32}, genes_new::Matrix{Float32}, mode::Symbol) :: Matrix{Float32}
    new_genes = Base.copy(genes_origianal)
    if mode == :d_sum
        new_genes += genes_new
        EnvironmentWrapper.normalize_genes!(new_genes)
    elseif mode == :min_0
        new_genes += genes_new
        EnvironmentWrapper.normalize_genes_min_0!(new_genes)
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