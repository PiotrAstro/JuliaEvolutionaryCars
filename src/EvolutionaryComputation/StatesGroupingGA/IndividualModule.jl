module IndividualModule

import ..Environment
import ..EnvironmentWrapper

import Random
import Plots
import Dates
import Logging
import Printf

export Individual, get_flattened_trajectories, get_id_track, get_fitness!, copy_individual, optimal_mixing_top_to_bottom!, optimal_mixing_bottom_to_top!, FIHC_flat!, FIHC_top_to_bottom!, save_decision_plot, clear_trajectory_memory!

global ID::Int = 0

mutable struct Individual
    genes::Vector{Int}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    time_tree::EnvironmentWrapper.TreeNode
    levels_trajectories::Vector{Vector{<:Environment.Trajectory}}
    _fitness::Float64
    _fitness_actual::Bool
    _verbose::Bool
    id_track::Int
end

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct, verbose::Bool=false)
    genes = Random.rand(1:EnvironmentWrapper.get_action_size(env_wrapper), EnvironmentWrapper.get_groups_number(env_wrapper))
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

function get_flattened_trajectories(individual::Individual) :: Vector{<:Environment.Trajectory}
    return vcat(individual.levels_trajectories...)
end

function get_flattened_trajectories(individuals::Vector{Individual}) :: Vector{<:Environment.Trajectory}
    return vcat([get_flattened_trajectories(individual) for individual in individuals]...)
end

function get_fitness!(individual::Individual) :: Float64
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
        individual.time_tree = EnvironmentWrapper.TreeNode(nothing, nothing, Vector{Int}())
    end
end

function mutate_flat!(individual::Individual, mutation_rate::Float64)
    action_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)
    for i in 1:length(individual.genes)
        if Random.rand() < mutation_rate
            individual.genes[i] = Random.rand(1:action_number)
            individual._fitness_actual = false
        end
    end    
end

function optimal_mixing_top_to_bottom_2_individuals(individual1::Individual, individual2) :: Individual
    new_individual = copy_individual(individual1)
    # print("\npre optimal mixing fitness: ", get_fitness!(new_individual))
    # root = individual.env_wrapper._similarity_tree
    root = new_individual.time_tree
    tree_levels = [[root.left, root.right]]

    individual1_genes = individual1.genes
    individual2_genes = get_other_genes(individual1, individual2, collect(1:length(individual1.genes)))

    # this one is test only!!!
    i = 1
    while i <= length(tree_levels)
        current_level = tree_levels[i]
        random_perm = Random.randperm(length(current_level))

        for node in current_level[random_perm]         
            old_elements = new_individual.genes[node.elements]
            old_fitness = get_fitness!(new_individual)

            if old_elements != individual1_genes[node.elements]
                new_individual.genes[node.elements] = individual1_genes[node.elements]
                new_individual._fitness_actual = false
            elseif old_elements != individual2_genes[node.elements]
                new_individual.genes[node.elements] = individual2_genes[node.elements]
                new_individual._fitness_actual = false
            end

            if get_fitness!(new_individual) > old_fitness
                # println("improvement from $old_fitness  to $(get_fitness!(new_individual))\ttree level $i")
                # save_decision_plot(individual)
            else
                new_individual.genes[node.elements] = old_elements
                new_individual._fitness = old_fitness
            end

            if !EnvironmentWrapper.is_leaf(node)
                if i == length(tree_levels)
                    push!(tree_levels, [node.left, node.right])
                else
                    push!(tree_levels[i+1], node.left, node.right)
                end
            end
        end

        i += 1
    end

    # print("\tpost optimal mixing fitness: $(get_fitness!(new_individual))\n")
    return new_individual
end

function optimal_mixing_top_to_bottom!(individual::Individual, other_individuals::Vector{Individual})
    if individual._verbose
        Logging.@info Printf.@sprintf("\npre optimal mixing fitness: %.2f\n", get_fitness!(individual))
    end

    # root = individual.env_wrapper._similarity_tree
    root = individual.time_tree
    tree_levels = [[root.left, root.right]]

    # this one is test only!!!
    i = 1
    while i <= length(tree_levels)
        current_level = tree_levels[i]
        random_perm = Random.randperm(length(current_level))

        for node in current_level[random_perm]
            # get one random individual
            # old_elements = individual.genes[node.elements]
            # old_fitness = get_fitness!(individual)
            # random_individual = other_individuals[Random.rand(1:length(other_individuals))]
            # if individual.genes[node.elements] != random_individual.genes[node.elements]
            #     individual.genes[node.elements] = random_individual.genes[node.elements]
            #     individual._fitness_actual = false
            #     new_fitness = get_fitness!(individual)

            #     if old_fitness >= new_fitness
            #         individual.genes[node.elements] = old_elements
            #         individual._fitness = old_fitness
            #     else
            #         println("improvement from $old_fitness to $new_fitness\ttree level $i")
            #         # save_decision_plot(individual)
            #     end
            # end
            individuals_copies = Vector{Individual}(undef, length(other_individuals))
            Threads.@threads for i in 1:length(other_individuals)
            # for i in 1:length(other_individuals)
                individual_tmp = copy_individual(individual)
                individuals_copies[i] = individual_tmp
                other_genes = get_other_genes(individual, other_individuals[i], node.elements)
                individual_tmp.genes[node.elements] = other_genes
                individual_tmp._fitness_actual = false
                get_fitness!(individual_tmp)
            end

            max_copy = argmax(get_fitness!, individuals_copies)
            if get_fitness!(max_copy) > get_fitness!(individual)
                individual.genes[node.elements] = max_copy.genes[node.elements]
                individual._fitness = max_copy._fitness
                if individual._verbose
                    Logging.@info Printf.@sprintf("improvement from %.2f  to %.2f\ttree level %d\n", get_fitness!(individual), get_fitness!(max_copy), i)
                end
                # save_decision_plot(individual)
            end

            if !EnvironmentWrapper.is_leaf(node)
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
        Logging.@info Printf.@sprintf("\tpost optimal mixing fitness: %.2f\n", get_fitness!(individual))
    end
end

function get_same_genes_percent(individual::Individual, other_individual::Individual) :: Float64
    other_individual_genes = get_other_genes(individual, other_individual, collect(1:length(individual.genes)))
    return sum(individual.genes .== other_individual_genes) / length(individual.genes)
end

function get_same_genes_percent(individual::Individual, other_individuals::Vector{Individual}) :: Vector{Float64}
    return [get_same_genes_percent(individual, other_individual) for other_individual in other_individuals]
end

function visualize(individual::Individual, visualization_env::Environment.AbstractEnvironment, visualization_kwargs::Dict{Symbol, Any})
    nn = EnvironmentWrapper.get_full_NN(individual.env_wrapper, individual.genes)
    Environment.visualize!(visualization_env, nn; visualization_kwargs...)
end

function get_other_genes(individual::Individual, other_individual::Individual, to_genes_indices::Vector{Int}) :: Vector{Int}
    return EnvironmentWrapper.translate(
        other_individual.env_wrapper,
        individual.env_wrapper,
        other_individual.genes,
        to_genes_indices
    )
end

function optimal_mixing_bottom_to_top!(individual::Individual, other_individuals::Vector{Individual})
    if individual._verbose
        Logging.@info Printf.@sprintf("\npre optimal mixing fitness: %.2f\n", get_fitness!(individual))
    end
    # root = individual.env_wrapper._similarity_tree
    root = individual.time_tree
    tree_levels = [[root.left, root.right]]

    i = 1
    while i <= length(tree_levels)
        current_level = tree_levels[i]
        for node in current_level
            if !EnvironmentWrapper.is_leaf(node)
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
            old_elements = individual.genes[node.elements]
            old_fitness = get_fitness!(individual)
            
            # get one random individual
            # random_individual = other_individuals[Random.rand(1:length(other_individuals))]
            # if individual.genes[node.elements] != random_individual.genes[node.elements]
            #     individual.genes[node.elements] = random_individual.genes[node.elements]
            #     individual._fitness_actual = false
            #     new_fitness = get_fitness!(individual)

            #     if old_fitness >= new_fitness
            #         individual.genes[node.elements] = old_elements
            #         individual._fitness = old_fitness
            #     else
            #         println("improvement from $old_fitness to $new_fitness\ttree level $i")
            #         # save_decision_plot(individual)
            #     end
            # end
            individuals_copies = Vector{Individual}(undef, length(other_individuals))
            Threads.@threads for i in 1:length(other_individuals)
            # for i in 1:length(other_individuals)
                individual_tmp = copy_individual(individual)
                individuals_copies[i] = individual_tmp
                other_genes = get_other_genes(individual, other_individuals[i], node.elements)
                individual_tmp.genes[node.elements] = other_genes
                individual_tmp._fitness_actual = false
                # FIHC_flat!(individual_tmp, node.elements)
                get_fitness!(individual_tmp)
            end

            max_copy = argmax(get_fitness!, individuals_copies)
            if get_fitness!(max_copy) > old_fitness
                individual.genes[node.elements] = max_copy.genes[node.elements]
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
end

function new_level_cosideration!(individual::Individual)
    trajectory, time_tree = EnvironmentWrapper.create_time_distance_tree(individual.env_wrapper, individual.genes)
    individual.time_tree = time_tree
    push!(individual.levels_trajectories, trajectory)
end

function FIHC_flat!(individual::Individual, indicies=collect(1:length(individual.genes)))
    # flat version
    # if individual._verbose
    #     Logging.@info Printf.@sprintf("\npre FIHC fitness: %.2f\n", get_fitness!(individual))
    # end
    random_perm = Random.shuffle(indicies)  # Random.randperm(length(individual.genes))
    actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)

    for gene_index in random_perm
        previous_fitness = get_fitness!(individual)
        previous_gene = individual.genes[gene_index]
        random_loci_order = [i for i in Random.randperm(actions_number) if i != previous_gene]

        individuals_copies = Vector{Individual}(undef, length(random_loci_order))
        Threads.@threads for i in eachindex(random_loci_order)
        # for i in eachindex(random_loci_order)
            individual_tmp = copy_individual(individual)
            individuals_copies[i] = individual_tmp
            individual_tmp.genes[gene_index] = random_loci_order[i]
            individual_tmp._fitness_actual = false
            get_fitness!(individual)
        end

        max_copy = argmax(get_fitness!, individuals_copies)
        if get_fitness!(max_copy) > previous_fitness
            individual.genes[gene_index] = max_copy.genes[gene_index]
            individual._fitness = max_copy._fitness
            # if individual._verbose
            #     Logging.@info Printf.@sprintf("improvement from %.2f  to %.2f\n", previous_fitness, get_fitness!(max_copy))
            # end
            # save_decision_plot(individual)
        end

        # for loci in random_loci_order
        #     individual.genes[gene_index] = loci
        #     individual._fitness_actual = false
        #     new_fitness = get_fitness!(individual)
        #     if new_fitness > previous_fitness
        #         print("improvement from $previous_fitness to $new_fitness\tgene index $gene_index\n")
        #     else
        #         individual.genes[gene_index] = previous_gene
        #         individual._fitness = previous_fitness
        #     end
        # end
    end

    # if individual._verbose
    #     Logging.@info Printf.@sprintf("\npost FIHC fitness: %.2f\n", get_fitness!(individual))
    # end
end

function FIHC_top_to_bottom!(individual::Individual)
    if individual._verbose
        Logging.@info Printf.@sprintf("\npre FIHC fitness: %.2f\n", get_fitness!(individual))
    end
    root = individual.env_wrapper._similarity_tree
    tree_levels = [[root.left, root.right]]
    actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)
    # individual.genes .= Random.rand(1:actions_number)
    # individual._fitness_actual = false
    # println(" FIHC initial fitness: ", get_fitness!(individual))
    overall_fitness_ckecks = 0

    # this one is test only!!!
    i = 1
    while i <= length(tree_levels)  # && i <= 15
        current_level = tree_levels[i]
        random_perm = Random.randperm(length(current_level))

        for node in current_level[random_perm]
            random_perm_actions = Random.randperm(actions_number)
            old_elements = individual.genes[node.elements]
            old_fitness = get_fitness!(individual)
            
            # for chosen_action in random_perm_actions
            #     individual.genes[node.elements] .= chosen_action
            #     individual._fitness_actual = false
            #     new_fitness = get_fitness!(individual)
            #     overall_fitness_ckecks += 1

            #     if old_fitness >= new_fitness
            #         individual.genes[node.elements] = old_elements
            #         individual._fitness = old_fitness
            #     else
            #         if individual._verbose
            #             Logging.@info Printf.@sprintf("improvement from %.2f  to %.2f\ttree level %d\n", old_fitness, new_fitness, i)
            #         end
            #         # save_decision_plot(individual)
            #         old_fitness = new_fitness
            #         old_elements = individual.genes[node.elements]
            #         break
            #     end
            # end

            individuals_copies = Vector{Individual}(undef, actions_number)
            Threads.@threads for chosen_action in random_perm_actions
                individual_tmp = copy_individual(individual)
                individuals_copies[chosen_action] = individual_tmp
                individual_tmp.genes[node.elements] .= chosen_action
                individual_tmp._fitness_actual = false
                get_fitness!(individual)
            end

            overall_fitness_ckecks += actions_number

            max_copy = argmax(get_fitness!, individuals_copies)
            if get_fitness!(max_copy) > old_fitness
                individual.genes[node.elements] = max_copy.genes[node.elements]
                individual._fitness = max_copy._fitness
                if individual._verbose
                    println("improvement from $old_fitness  to $(get_fitness!(max_copy))\ttree level $i")
                end
                # save_decision_plot(individual)
            end

            if !EnvironmentWrapper.is_leaf(node)
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
end

function save_decision_plot(individual::Individual, path::Union{String, Nothing}=nothing)
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




# ------------------------------------------------------------------------------------------
# Old functions

# function mutate_top_to_bottom!(individual::Individual, mutation_rate::Float64)
#     root = individual.env_wrapper._similarity_tree
#     tree_levels = [[root.left, root.right]]
#     actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)

#     i = 1
#     while i <= length(tree_levels)
#         current_level = tree_levels[i]
#         random_perm = Random.randperm(length(current_level))

#         for node in current_level[random_perm]
#             if Random.rand() < mutation_rate
#                 individual.genes[node.elements] .= Random.rand(1:actions_number)
#                 individual._fitness_actual = false
#             end

#             if !EnvironmentWrapper.is_leaf(node)
#                 if i == length(tree_levels)
#                     push!(tree_levels, [node.left, node.right])
#                 else
#                     push!(tree_levels[i+1], node.left, node.right)
#                 end
#             end
#         end

#         i += 1
#     end
# end

# function actualize_genes!(individual::Individual, new_genes::Vector{Int})
#     individual.genes = new_genes
#     individual._fitness_actual = false
# end

# function crossover(individual1::Individual, individual2::Individual) :: Tuple{Individual, Individual}
#     new_individual1 = copy_individual(individual1)
#     new_individual2 = copy_individual(individual2)
#     for i in 1:length(new_individual1.genes)
#         if Random.rand() < 0.5
#             new_individual1.genes[i] = individual2.genes[i]
#             new_individual2.genes[i] = individual1.genes[i]
#         end
#     end

#     new_individual1._fitness_actual = false
#     new_individual2._fitness_actual = false

#     return (new_individual1, new_individual2)
# end

end