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
    evals = 0

    if strategy == :one_rand
        other = other_individuals[rand(1:other_n)]
        for nodes_level in genes_comb
            for node in Random.shuffle(nodes_level)
                accept_if_better!(ind, other, node)
                evals += 1
            end
        end
    elseif strategy == :one_tournament
        other_inds = other_individuals[Random.randperm(other_n)[1:2]]
        other = other_inds[1]
        if Random.rand() < 0.5
            other = get_fitness!(other_inds[1]) > get_fitness!(other_inds[2]) ? other_inds[1] : other_inds[2]
        end

        for nodes_level in genes_comb
            for node in Random.shuffle(nodes_level)
                accept_if_better!(ind, other, node)
                evals += 1
            end
        end
    elseif strategy == :all_seq
        for other in other_individuals
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
                for other in other_individuals
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
    # for _ in 1:10
    #     FIHC_test!(ind; ind.fihc_dict...)
    # end

    # for _ in 1:10
    #     FIHC_test!(other; other.fihc_dict...)
    # end

    # println("\n\n\nThis:")
    # display(ind.genes)
    # println("\nOther:")
    # display(EnvironmentWrapper.translate(other.env_wrapper, other.genes, ind.env_wrapper))
    # throw("dsdsvdsfvfdbjkfd")

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

function get_genes_combinations(ind::Individual, mode::Symbol)::Vector{Vector{Vector{Float32}}}
    genes_n = EnvironmentWrapper.get_groups_number(ind.env_wrapper)

    if mode == :all
        return [[ones(Float32, genes_n)]]
    elseif mode == :flat
        arr = [[zeros(Float32, genes_n) for _ in 1:genes_n]]
        for i in 1:genes_n
            arr[1][i][i] .= 1.0f0
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
    total_evaluations::Int
    fihc::Dict
    cross::Dict
    initial_genes_mode::Symbol
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
    individuals_n::Int,
    fihc::Dict,
    cross::Dict,
    initial_genes_mode::Symbol
)
    environment_type = Environment.get_environment(environment)
    environments = [(environment_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]
    visualization_env = (environment_type)(;environment_visualization_kwargs...)
    

    env_wrapper_struct = EnvironmentWrapper.EnvironmentWrapperStruct(
        environments;
        env_wrapper...
    )

    individuals = [Individual(env_wrapper_struct, cross, fihc, initial_genes_mode) for _ in 1:individuals_n]

    # Threads.@threads for ind in individuals
    for ind in individuals
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
        0,
        fihc,
        cross,
        initial_genes_mode,
        false
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

    # (generation, total_evaluations, best_fitness)
    list_with_results = Vector{Tuple{Int, Int, Float64}}()

    for generation in 1:max_generations
        if csgs.total_evaluations >= max_evaluations
            break
        end

        Base.GC.gc(true)  # we want to reclaim all possible memory, true means we will do full GC

        start_time = time()

        individuals_copy_for_crossover = [individual_copy(ind) for ind in csgs.population]
        # new_env_wrapper = Threads.@spawn EnvironmentWrapper.create_new_based_on(
        new_env_wrapper = EnvironmentWrapper.create_new_based_on(
            csgs.current_env_wrapper,
            [
                (1.0, get_flattened_trajectories(individuals_copy_for_crossover)),
            ]
        )
        # new_individuals_evals = [(Threads.@spawn run_one_individual_generation(ind, individuals_copy_for_crossover)) for ind in csgs.population]
        new_individuals_evals = [run_one_individual_generation(ind, individuals_copy_for_crossover) for ind in csgs.population]
        for i in eachindex(csgs.population)
            csgs.total_evaluations += new_individuals_evals[i]
            # csgs.total_evaluations += fetch(new_individuals_evals[i])
        end
        csgs.current_env_wrapper = new_env_wrapper
        # csgs.current_env_wrapper = fetch(new_env_wrapper)
        best_ind_arg = argmax(get_fitness!.(csgs.population))
        csgs.best_individual = individual_copy(csgs.population[best_ind_arg])

        # put random individual at random place different from best individual
        random_ind = rand(collect(eachindex(csgs.population))[eachindex(csgs.population) .!= best_ind_arg])
        csgs.population[random_ind] = Individual(csgs.current_env_wrapper, csgs.cross, csgs.fihc, csgs.initial_genes_mode, csgs.verbose)
        get_fitness!(csgs.population[random_ind])

        end_time = time()

        # if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
        #     Ind.visualize(p3.best_individual, p3.visualization_env, p3.visualization_kwargs)
        # end

        best_fitness = get_fitness!(csgs.best_individual)
        if log
            fitnesses = get_fitness!.(csgs.population)
            quantiles = [0.25, 0.5, 0.75, 0.95]
            quantiles_values = Statistics.quantile(fitnesses, quantiles)
            mean_fitness = Statistics.mean(fitnesses)
            elapsed_time = end_time - start_time
            Logging.@info "\n\n\n\n\n\nGeneration $generation\nTotal evaluations: $(csgs.total_evaluations)\n" *
            Printf.@sprintf("elapsed_time: %.2f s\nbest_fitness: %.2f\nmean_fitness: %.2f", elapsed_time, best_fitness, mean_fitness) *
            "quantiles:   $(join([(Printf.@sprintf "%.2f: %.2f" quantile fitness) for (quantile, fitness) in zip(quantiles, quantiles_values)], "   "))\n"
        end
        
        push!(
            list_with_results,
            (generation, csgs.total_evaluations, best_fitness)
        )
    end

    data_frame = DataFrames.DataFrame(
        list_with_results,
        [:generation, :total_evaluations, :best_fitness]
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
function run_one_individual_generation(ind::Individual, other::Vector{Individual})::Int
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
    return evaluations
end





# --------------------------------------------------------------------------------------------------
# tests

function run_test(csgs::ContinuousStatesGroupingSimpleGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, fihc_settings::Dict)
    new_ind = Individual(csgs.current_env_wrapper, csgs.initial_genes_mode, log)
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
    hier_factor::Float64
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

    if fihc_mode == :per_gene_rand
        evals = 0
        for nodes_level in ind.env_wrapper._distance_membership_levels
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
        return evals
    
    elseif fihc_mode == :hier_decrease
        evals = 0
        factor_tmp = factor
        for nodes_level in ind.env_wrapper._distance_membership_levels
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
        return evals

    elseif fihc_mode == :hier_increase
        evals = 0
        genes_combinations = ind.env_wrapper._distance_membership_levels
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
        return evals

    # elseif fihc_mode == :disc_fihc
    #     for col in eachcol(ind.genes)
    #         argmax_col = argmax(col)
    #         col .= 0.0
    #         col[argmax_col] = 1.0
    #     end

    #     evals = 0
    #     actions_n = EnvironmentWrapper.get_action_size(ind.env_wrapper)
    #     for nodes_level in ind.env_wrapper._distance_membership_levels
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
    #     return evals

    # elseif fihc_mode == :fihc_cont
    #     evals = 0
    #     actions_n = EnvironmentWrapper.get_action_size(ind.env_wrapper)
    #     for nodes_level in ind.env_wrapper._distance_membership_levels
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
    #     return evals

    else
        throw(ArgumentError("Unknown mode: $fihc_mode"))
    end
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