module IndividualModule

import ..Environment
import ..EnvironmentWrapper
import ..NeuralNetwork
import ..StatesGrouping

import Random
import Plots
import Dates
import Logging
import Printf
import Statistics
import Distances

export Individual, get_flattened_trajectories, get_id_track, get_fitness!, copy_individual, optimal_mixing_bottom_to_top!, FIHC_flat!, FIHC_top_to_bottom!, save_decision_plot, clear_trajectory_memory!, default_FIHC!, get_avg_kld

global ID::Int = 0

mutable struct Individual
    genes::Matrix{Float32}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    time_tree::StatesGrouping.TreeNode
    levels_trajectories::Vector{Vector{<:Environment.Trajectory}}
    _fitness::Float64
    _fitness_actual::Bool
    _verbose::Bool
    id_track::Int
end

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, verbose::Bool=false)
    genes = EnvironmentWrapper.new_genes(env_wrapper)
    fitness = -Inf
    fitness_actual = false
    trajectories, time_tree = EnvironmentWrapper.create_time_distance_tree(env_wrapper, genes)
    global ID += 1
    return Individual(
        genes,
        env_wrapper,
        time_tree,
        [trajectories],
        fitness,
        fitness_actual,
        verbose,
        ID
    )
end

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    return Individual(env_wrapper, EnvironmentWrapper.is_verbose(env_wrapper))
end

function get_id_track(individual::Individual)
    return individual.id_track
end

function get_flattened_trajectories(individual::Individual)::Vector{<:Environment.Trajectory}
    return vcat(individual.levels_trajectories...)
end

function get_flattened_trajectories(individuals::Vector{Individual})::Vector{<:Environment.Trajectory}
    return vcat([get_flattened_trajectories(individual) for individual in individuals]...)
end

function get_avg_kld(individual::Individual, other_individuals::AbstractVector{Individual})::Float64
    my_genes = Environment.get_genes(individual.env_wrapper, individual.genes)
    other_genes = [Environment.translate(individual.env_wrapper, other_individual.env_wrapper, other_individual.genes) for other_individual in other_individuals]
    return Statistics.mean(
        vec([
            Distances.kl_divergence(@view(my_genes[:, exemplar_responded]), @view(other_genes[i][:, exemplar_responded]))
            for exemplar_responded in axes(my_genes, 2), i in eachindex(other_individuals)
        ])
    )
end

function get_fitness!(individual::Individual)::Float64
    if !individual._fitness_actual
        individual._fitness = EnvironmentWrapper.get_fitness(individual.env_wrapper, individual.genes)
        individual._fitness_actual = true
    end

    return individual._fitness
end

function copy_individual(individual::Individual)
    return Individual(
        copy(individual.genes),
        individual.env_wrapper,
        individual.time_tree,
        [trajectories for trajectories in individual.levels_trajectories],
        individual._fitness,
        individual._fitness_actual,
        individual._verbose,
        individual.id_track
    )
end

function clear_trajectory_memory!(individual::Individual)
    if !isempty(individual.levels_trajectories)
        type_trajectories_levels = typeof(individual.levels_trajectories)
        individual.levels_trajectories = type_trajectories_levels()
        individual.time_tree = StatesGrouping.TreeNode(nothing, nothing, Vector{Int}())
    end
end

function change_genes_at_position!(individual::Individual, positions::Vector{Int}, action_index::Int)
    individual.genes[action_index, positions] .+= 1.0f0
    EnvironmentWrapper.normalize_genes!(individual.genes)
end

function visualize(individual::Individual, visualization_env::Environment.AbstractEnvironment, visualization_kwargs::Dict{Symbol,Any})
    nn = EnvironmentWrapper.get_full_NN(individual.env_wrapper, individual.genes)
    Environment.visualize!(visualization_env, nn; visualization_kwargs...)
end

function get_other_genes(individual::Individual, other_individual::Individual, to_genes_indices::Vector{Int})::Matrix{Float32}
    return EnvironmentWrapper.translate(
        other_individual.env_wrapper,
        other_individual.genes,
        individual.env_wrapper,
        to_genes_indices
    )
end

"""
Returns number of total evaluations
"""
function optimal_mixing_bottom_to_top!(individual::Individual, other_individuals::Vector{Individual})::Int
    if individual._verbose
        Logging.@info Printf.@sprintf("\npre optimal mixing fitness: %.2f\n", get_fitness!(individual))
    end
    # root = individual.env_wrapper._similarity_tree
    root = individual.time_tree
    tree_levels = [[root.left, root.right]]
    total_evaluations = 0

    i = 1
    while i <= length(tree_levels)
        current_level = tree_levels[i]
        for node in current_level
            if !StatesGrouping.is_leaf(node)
                if i == length(tree_levels)
                    push!(tree_levels, [node.left, node.right])
                else
                    push!(tree_levels[i+1], node.left, node.right)
                end
            end
        end

        i += 1
    end

    i = length(tree_levels)
    while i >= 1
        current_level = tree_levels[i]
        random_perm = Random.randperm(length(current_level))

        for node in current_level[random_perm]
            total_evaluations += length(other_individuals)
            old_fitness = get_fitness!(individual)

            individuals_copies = Vector{Individual}(undef, length(other_individuals))
            Threads.@threads for i in 1:length(other_individuals)
                # for i in 1:length(other_individuals)
                individual_tmp = copy_individual(individual)
                individuals_copies[i] = individual_tmp
                other_genes = get_other_genes(individual, other_individuals[i], node.elements)
                individual_tmp.genes[:, node.elements] = other_genes
                individual_tmp._fitness_actual = false
                get_fitness!(individual_tmp)
            end

            max_copy = argmax(get_fitness!, individuals_copies)
            if get_fitness!(max_copy) > old_fitness
                individual.genes = max_copy.genes
                individual._fitness = max_copy._fitness
                if individual._verbose
                    Logging.@info Printf.@sprintf("improvement from %.2f  to %.2f\ttree level %d\n", old_fitness, get_fitness!(max_copy), i)
                end
                # save_decision_plot(individual)
            end
        end
        i -= 1
    end

    if individual._verbose
        Logging.@info Printf.@sprintf("\tpost optimal mixing fitness: %.2f\n", get_fitness!(individual))
    end

    return total_evaluations
end

function new_level_cosideration!(individual::Individual)
    trajectory, time_tree = EnvironmentWrapper.create_time_distance_tree(individual.env_wrapper, individual.genes)
    individual.time_tree = time_tree
    push!(individual.levels_trajectories, trajectory)
end

function default_FIHC!(individual::Individual)
    FIHC_top_to_bottom!(individual)
end

"""
returns number of fitness checks
"""
function FIHC_flat!(individual::Individual, indicies=collect(1:size(individual.genes, 2)))::Int
    # flat version
    if individual._verbose
        Logging.@info Printf.@sprintf("\npre FIHC fitness: %.2f\n", get_fitness!(individual))
    end
    random_perm = Random.shuffle(indicies)
    actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)

    for gene_index in random_perm
        previous_fitness = get_fitness!(individual)
        random_loci_order = Random.randperm(actions_number)

        individuals_copies = Vector{Individual}(undef, length(random_loci_order))
        Threads.@threads for i in eachindex(random_loci_order)
            # for i in eachindex(random_loci_order)
            individual_tmp = copy_individual(individual)
            individuals_copies[i] = individual_tmp
            change_genes_at_position!(individual_tmp, [gene_index], random_loci_order[i])
            individual_tmp._fitness_actual = false
            get_fitness!(individual)
        end

        max_copy = argmax(get_fitness!, individuals_copies)
        if get_fitness!(max_copy) > previous_fitness
            individual.genes = max_copy.genes
            individual._fitness = max_copy._fitness
            if individual._verbose
                Logging.@info Printf.@sprintf("improvement from %.2f  to %.2f\n", previous_fitness, get_fitness!(max_copy))
            end
            # save_decision_plot(individual)
        end
    end

    if individual._verbose
        Logging.@info Printf.@sprintf("\npost FIHC fitness: %.2f\n", get_fitness!(individual))
    end

    return length(indicies) * actions_number
end

"""
returns number of fitness checks
"""
function FIHC_top_to_bottom!(individual::Individual)::Int
    if individual._verbose
        Logging.@info Printf.@sprintf("\npre FIHC fitness: %.2f\n", get_fitness!(individual))
    end
    root = individual.env_wrapper._similarity_tree
    tree_levels = [[root.left, root.right]]
    actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)
    overall_fitness_ckecks = 0

    i = 1
    while i <= length(tree_levels)  # && i <= 15
        current_level = tree_levels[i]
        random_perm = Random.randperm(length(current_level))

        for node in current_level[random_perm]
            random_perm_actions = Random.randperm(actions_number)
            old_fitness = get_fitness!(individual)

            individuals_copies = Vector{Individual}(undef, actions_number)
            Threads.@threads for chosen_action in random_perm_actions
                individual_tmp = copy_individual(individual)
                individuals_copies[chosen_action] = individual_tmp
                change_genes_at_position!(individual_tmp, node.elements, chosen_action)
                individual_tmp._fitness_actual = false
                get_fitness!(individual)
            end

            overall_fitness_ckecks += actions_number

            max_copy = argmax(get_fitness!, individuals_copies)
            if get_fitness!(max_copy) > old_fitness
                individual.genes = max_copy.genes
                individual._fitness = max_copy._fitness
                if individual._verbose
                    println("improvement from $old_fitness  to $(get_fitness!(max_copy))\ttree level $i")
                end
                # save_decision_plot(individual)
            end

            if !StatesGrouping.is_leaf(node)
                if i == length(tree_levels)
                    push!(tree_levels, [node.left, node.right])
                else
                    push!(tree_levels[i+1], node.left, node.right)
                end
            end
        end

        i += 1
    end

    if individual._verbose
        Logging.@info Printf.@sprintf("\toverall_fitness_ckecks: %d", overall_fitness_ckecks) * Printf.@sprintf("\tpost FIHC fitness: %.2f", get_fitness!(individual))
    end

    return overall_fitness_ckecks
end

function save_decision_plot(individual::Individual, path::Union{String,Nothing}=nothing)
    env_wrapper = individual.env_wrapper
    action_number = EnvironmentWrapper.get_action_size(env_wrapper)
    genes_grouped = [[id for id in eachindex(individual.genes) if individual.genes[id] == action] for action in 1:action_number]

    Plots.scatter(env_wrapper._encoded_exemplars[1, genes_grouped[1]], env_wrapper._encoded_exemplars[2, genes_grouped[1]], legend=false, size=(1500, 1500), markerstrokewidth=0)
    for i in 2:action_number
        Plots.scatter!(env_wrapper._encoded_exemplars[1, genes_grouped[i]], env_wrapper._encoded_exemplars[2, genes_grouped[i]], legend=false, markerstrokewidth=0)
    end

    if isnothing(path)
        timestamp_string = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        path = "log/_P3_FIHC_output/$(timestamp_string)_$(get_fitness!(individual)).png"
    end

    Plots.savefig(path)
end

end