module ContinuousStatesGroupingSimpleGA

import DataFrames
import Logging
import Printf
import Random
import Statistics

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
    _trajectories::Vector{<:Environment.Trajectory}
    _trajectories_actual::Bool
    _fitness::Float64
    _fitness_actual::Bool
    _verbose::Bool
end

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, verbose=false)
    genes = EnvironmentWrapper.new_genes(env_wrapper)
    return Individual(
        genes,
        env_wrapper,
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
        ind._trajectories,
        ind._trajectories_actual,
        ind._fitness,
        ind._fitness_actual,
        ind._verbose
    )
end

function get_flattened_trajectories(individuals::Vector{Individual})::Vector{<:Environment.Trajectory}
    return vcat([individual._trajectories for individual in individuals]...)
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
        individual._trajectories_actual = true
        individual._fitness_actual = true
        individual._fitness = sum([tra.rewards_sum for tra in individual._trajectories])
    end

    return individual._fitness
end

function FIHC_crossover!(ind::Individual, other_individuals::Vector{Individual}) :: Int
    permutation_of_individuals = Random.randperm(length(other_individuals))
    did_anything_change = false

    for ind_other in other_individuals[permutation_of_individuals]
        old_genes = Base.copy(ind.genes)
        old_fitness = get_fitness!(ind)

        ind.genes += ind_other.genes
        EnvironmentWrapper.normalize_genes_min_0!(ind.genes)
        ind._fitness_actual = false
        new_fitness = get_fitness!(ind)

        if new_fitness < old_fitness
            ind.genes = old_genes
            ind._fitness = old_fitness
        else
            did_anything_change = true
        end
    end

    if did_anything_change
        ind._trajectories_actual = false
    end

    return length(other_individuals)
end

function FIHC!(ind::Individual, check_max_n_combinations::Int = 1) :: Int
    genes_n = size(ind.genes, 2)
    permutation_of_genes = Random.randperm(genes_n)
    did_anything_change = false

    for gene_id in permutation_of_genes
        old_genes = Base.copy(ind.genes)
        old_fitness = get_fitness!(ind)

        random_new_gene = rand(Float32, size(ind.genes, 1))
        ind._fitness_actual = false
        ind.genes[:, gene_id] += random_new_gene
        EnvironmentWrapper.normalize_genes_min_0!(ind.genes)
        new_fitness = get_fitness!(ind)
        if new_fitness < old_fitness
            ind.genes = old_genes
            ind._fitness = old_fitness
        else
            did_anything_change = true
        end
    end

    if did_anything_change
        ind._trajectories_actual = false
    end

    return genes_n
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
    fihc::Dict
)
    environment_type = Environment.get_environment(environment)
    environments = [(environment_type)(;environment_kwarg...) for environment_kwarg in environment_kwargs]
    visualization_env = (environment_type)(;environment_visualization_kwargs...)
    

    env_wrapper_struct = EnvironmentWrapper.EnvironmentWrapperStruct(
        environments;
        env_wrapper...
    )

    individuals = [Individual(env_wrapper_struct) for _ in 1:individuals_n]

    Threads.@threads for ind in individuals
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
        false
    )
end

function AbstractOptimizerModule.run!(csgs::ContinuousStatesGroupingSimpleGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, visualize_each_n_epochs::Int=0) :: DataFrames.DataFrame
    # --------------------------------------------------
    # Test!!!
    return run_test(csgs; max_generations=max_generations, max_evaluations=max_evaluations, log=log, fihc_settings=csgs.fihc)

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
        new_env_wrapper = Threads.@spawn EnvironmentWrapper.create_new_based_on(
            csgs.current_env_wrapper,
            [
                (1.0, get_flattened_trajectories(individuals_copy_for_crossover)),
            ]
        )
        new_individuals_evals = [Threads.@spawn run_one_individual_generation(ind, individuals_copy_for_crossover) for ind in csgs.population]
        for i in eachindex(csgs.population)
            csgs.total_evaluations += fetch(new_individuals_evals[i])
        end
        csgs.current_env_wrapper = fetch(new_env_wrapper)
        best_ind_arg = argmax([get_fitness!(ind) for ind in csgs.population])
        csgs.best_individual = individual_copy(csgs.population[best_ind_arg])

        # put random individual at random place different from best individual
        random_ind = rand(collect(eachindex(csgs.population))[eachindex(csgs.population) .!= best_ind_arg])
        csgs.population[random_ind] = Individual(csgs.current_env_wrapper, csgs.verbose)
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

function run_one_individual_generation(ind::Individual, other::Vector{Individual})::Int
    evaluations = FIHC_crossover!(ind, other)
    evaluations += FIHC!(ind)
    get_trajectories!(ind)
    return evaluations
end





# --------------------------------------------------------------------------------------------------
# tests

function run_test(csgs::ContinuousStatesGroupingSimpleGA_Algorithm; max_generations::Int, max_evaluations::Int, log::Bool, fihc_settings::Dict)
    new_ind = Individual(csgs.current_env_wrapper)
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
    if ind._fitness < 480.0
        get_fitness!(ind)
    else
        return ind._fitness
    end
end

function FIHC_test!(ind::Individual;
    fihc_mode::Symbol,
    genes_combination::Symbol,
    norm_mode::Symbol,
    random_matrix_mode::Symbol,
    factor::Float64,
) :: Int

    if fihc_mode == :matrix_rand
        evals_num_now = EnvironmentWrapper.get_groups_number(ind.env_wrapper)  # I want to run few more so that I do not hav too many entries in logs
        for _ in evals_num_now
            old_fitness = get_fitness_test!(ind)
            old_genes = Base.copy(ind.genes)
            new_genes = ind.genes .+ generate_random_matrix(size(ind.genes, 1), size(ind.genes, 2), factor, random_matrix_mode)
            ind._fitness_actual = false
            ind.genes = norm_genes(new_genes, norm_mode)
            
            if get_fitness_test!(ind) < old_fitness
                ind.genes = old_genes
                ind._fitness = old_fitness
            end
        end
        return evals_num_now

    elseif fihc_mode == :per_gene_rand
        evals = 0
        for nodes_level in get_genes_combination(ind.env_wrapper, genes_combination)
            for node in nodes_level
                old_fitness = get_fitness_test!(ind)
                old_genes = Base.copy(ind.genes)
                new_genes = Base.copy(ind.genes)
                new_genes[:, node] .+= generate_random_matrix(size(ind.genes, 1), length(node), factor, random_matrix_mode)
                ind._fitness_actual = false
                ind.genes = norm_genes(new_genes, norm_mode)

                evals += 1
                
                if get_fitness_test!(ind) < old_fitness
                    ind.genes = old_genes
                    ind._fitness = old_fitness
                end
            end
        end
        return evals

    elseif fihc_mode == :disc_fihc
        for col in eachcol(ind.genes)
            argmax_col = argmax(col)
            col .= 0.0
            col[argmax_col] = 1.0
        end

        evals = 0
        actions_n = EnvironmentWrapper.get_action_size(ind.env_wrapper)
        for nodes_level in get_genes_combination(ind.env_wrapper, genes_combination)
            for node in nodes_level
                for action in 1:actions_n
                    old_fitness = get_fitness_test!(ind)
                    old_genes = Base.copy(ind.genes)
                    new_genes = Base.copy(ind.genes)
                    new_genes[:, node] .= 0.0
                    new_genes[action, node] = 1.0
                    ind._fitness_actual = false
                    ind.genes = new_genes

                    evals += 1
                    
                    if get_fitness_test!(ind) < old_fitness
                        ind.genes = old_genes
                        ind._fitness = old_fitness
                    end
                end
            end
        end
        return evals

    elseif fihc_mode == :fihc_cont
        evals = 0
        actions_n = EnvironmentWrapper.get_action_size(ind.env_wrapper)
        for nodes_level in get_genes_combination(ind.env_wrapper, genes_combination)
            for node in nodes_level
                for action in 1:actions_n
                    old_fitness = get_fitness_test!(ind)
                    old_genes = Base.copy(ind.genes)
                    new_genes = Base.copy(ind.genes)
                    new_genes[action, node] .+= factor
                    ind._fitness_actual = false
                    ind.genes = norm_genes(new_genes, norm_mode)

                    evals += 1
                    
                    if get_fitness_test!(ind) < old_fitness
                        ind.genes = old_genes
                        ind._fitness = old_fitness
                    end
                end
            end
        end
        return evals

    else
        throw(ArgumentError("Unknown mode: $fihc_mode"))
    end
end

function get_genes_combination(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, mode::Symbol) :: Vector{Vector{Vector{Int}}}
    genes_n = EnvironmentWrapper.get_groups_number(env_wrapper)
    if mode == :flat
        return [[[i] for i in Random.randperm(genes_n)]]
    elseif mode == :hier
        tree = env_wrapper._similarity_tree
        current_nodes = [tree.left, tree.right]
        result = Vector{Vector{Vector{Int}}}()
        while !isempty(current_nodes)
            push!(result, [current_nodes[i].elements for i in Random.randperm(length(current_nodes))])
            new_nodes = []
            for node in current_nodes
                if !isnothing(node.left)
                    push!(new_nodes, node.left)
                end
                if !isnothing(node.right)
                    push!(new_nodes, node.right)
                end
            end
            current_nodes = new_nodes
        end
        return result
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end

function norm_genes(genes::Matrix{Float32}, mode::Symbol) :: Matrix{Float32}
    new_genes = Base.copy(genes)
    if mode == :d_sum
        EnvironmentWrapper.normalize_genes!(new_genes)
    elseif mode == :min_0
        EnvironmentWrapper.normalize_genes_min_0!(new_genes)
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
    return new_genes
end

function generate_random_matrix(latent_space::Int, n_clusters::Int, factor::Float64, mode::Symbol) :: Matrix{Float32}
    if mode == :rand
        return rand(Float32, latent_space, n_clusters) .* factor
    elseif mode == :randn
        return randn(Float32, latent_space, n_clusters) .* factor
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end


end # module ContinuousStatesGroupingP3