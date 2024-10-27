module NormalGA

import ..NeuralNetwork
import ..Environment

import ..EnvironmentWrapper

import Clustering
import Random
import StatsBase
import Plots
import Dates

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct Individual
    genes::Vector{Int}
    const env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    _fitness::Float64
    _fitness_actual::Bool
end

mutable struct GA_Struct
    _population::Vector{Individual}
    # _dependencies_matrices::Vector{Array{Int, 4}}
    best_individual::Individual
    const env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
end

# --------------------------------------------------------------------------------------------------
# Public functions

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    genes = Random.rand(1:EnvironmentWrapper.get_action_size(env_wrapper), EnvironmentWrapper.get_groups_number(env_wrapper))
    fitness = -Inf
    fitness_actual = false
    return Individual(genes, env_wrapper, fitness, fitness_actual)
end

function GA_Struct(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    individuals = [Individual(env_wrapper) for _ in 1:100]
    for individual in individuals
        get_fitness!(individual)
    end
    return GA_Struct(individuals, individuals[1], env_wrapper)
end

function get_fitness!(individual::Individual) :: Float64
    if !individual._fitness_actual
        individual._fitness = EnvironmentWrapper.get_fitness(individual.env_wrapper, individual.genes)
        individual._fitness_actual = true
    end

    return individual._fitness
end

function copy_individual(individual::Individual)
    return Individual(copy(individual.genes), individual.env_wrapper, individual._fitness, individual._fitness_actual)
end

function crossover(individual1::Individual, individual2::Individual) :: Tuple{Individual, Individual}
    new_individual1 = copy_individual(individual1)
    new_individual2 = copy_individual(individual2)
    for i in 1:length(new_individual1.genes)
        if Random.rand() < 0.5
            new_individual1.genes[i] = individual2.genes[i]
            new_individual2.genes[i] = individual1.genes[i]
        end
    end

    new_individual1._fitness_actual = false
    new_individual2._fitness_actual = false

    return (new_individual1, new_individual2)
end

function optimal_mixing(individual1::Individual, individual2::Individual)
    # print("\npre optimal mixing fitness: $(get_fitness!(individual1))  $(get_fitness!(individual2))")
    individual = copy_individual(individual1)
    root = individual.env_wrapper._similarity_tree
    tree_levels = [[root.left, root.right]]

    # this one is test only!!!
    i = 1
    while i <= length(tree_levels) && i <= 5
        current_level = tree_levels[i]
        random_perm = Random.randperm(length(current_level))

        for node in current_level[random_perm]         
            old_elements = individual.genes[node.elements]
            old_fitness = get_fitness!(individual)
            
            # get one random individual
            giver1_elements = individual1.genes[node.elements]
            giver2_elements = individual2.genes[node.elements]
            giver_elements = nothing
            if giver1_elements != giver2_elements
                if giver1_elements != old_elements
                    giver_elements = giver1_elements
                else
                    giver_elements = giver2_elements
                end
            end

            if !isnothing(giver_elements)
                individual.genes[node.elements] = giver_elements
                individual._fitness_actual = false
                new_fitness = get_fitness!(individual)

                if old_fitness >= new_fitness
                    individual.genes[node.elements] = old_elements
                    individual._fitness = old_fitness
                else
                    # println("improvement from $old_fitness to $new_fitness\ttree level $i")
                    # save_decision_plot(individual)
                    old_fitness = new_fitness
                    old_elements = individual.genes[node.elements]
                end
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

    # print("\tpost optimal mixing fitness: $(get_fitness!(individual))\n")
    return individual
end

function mutate!(individual::Individual, probability::Float64)
    action_size = EnvironmentWrapper.get_action_size(individual.env_wrapper)
    for i in 1:length(individual.genes)
        if Random.rand() < probability
            individual.genes[i] = Random.rand(1:action_size)
        end
    end
    individual._fitness_actual = false
end

function tournament_selection(ga::GA_Struct) :: Tuple{Individual, Individual}
    # select 4 random distinct individuals
    individuals = StatsBase.sample(ga._population, 4, replace=false)
    individual1 = individuals[1]
    if Random.rand() < 0.5
        if Random.rand() < 0.5
            individual1 = individuals[2]
        end
    elseif get_fitness!(individuals[2]) > get_fitness!(individual1)
        individual1 = individuals[2]
    end

    individual2 = individuals[3]
    if Random.rand() < 0.5
        if Random.rand() < 0.5
            individual2 = individuals[4]
        end
    elseif get_fitness!(individuals[4]) > get_fitness!(individual2)
        individual2 = individuals[4]
    end

    return (individual1, individual2)
end

function _check_new_best!(ga::GA_Struct)
    for individual in ga._population
        if get_fitness!(individual) > get_fitness!(ga.best_individual)
            FIHC!(individual)
            ga.best_individual = copy_individual(individual)

            # if get_fitness!(individual) > 480.0
                save_decision_plot(individual)
            # end
        end
    end
end

function save_decision_plot(individual::Individual)
    env_wrapper = individual.env_wrapper
    action_number = EnvironmentWrapper.get_action_size(env_wrapper)
    genes_grouped = [[id for id in eachindex(individual.genes) if individual.genes[id] == action] for action in 1:action_number]
    
    Plots.scatter(env_wrapper._encoded_exemplars[1, genes_grouped[1]], env_wrapper._encoded_exemplars[2, genes_grouped[1]], legend=false, size=(1500, 1500), markerstrokewidth=0)
    for i in 2:action_number
        Plots.scatter!(env_wrapper._encoded_exemplars[1, genes_grouped[i]], env_wrapper._encoded_exemplars[2, genes_grouped[i]], legend=false, markerstrokewidth=0)
    end

    timestamp_string = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    Plots.savefig("log/individual_genotype_$(timestamp_string)_$(get_fitness!(individual)).png")
end

function generation!(ga::GA_Struct)
    for i in 1:1
        ga._population[i] = Individual(ga.env_wrapper)
    end
    new_population = Vector{Individual}(undef, length(ga._population))
    Threads.@threads for i in 1:2:length(ga._population)
    # for i in 1:2:length(ga._population)
        # individual1, individual2 = tournament_selection(ga)
        # new_individual1, new_individual2 = crossover(individual1, individual2)
        # mutate!(new_individual1, 0.05)
        # mutate!(new_individual2, 0.05)

        # original = get_fitness!(individual1) > get_fitness!(individual2) ? individual1 : individual2
        # new_individual = get_fitness!(new_individual1) > get_fitness!(new_individual2) ? new_individual1 : new_individual2
        # new_population[i] = original
        # new_population[i + 1] = new_individual

        individual1, individual2 = tournament_selection(ga)
        original = get_fitness!(individual1) > get_fitness!(individual2) ? individual1 : individual2
        new_individual = optimal_mixing(individual1, individual2)
        new_population[i] = original
        new_population[i + 1] = new_individual

        # println("parents fitness: ", get_fitness!(individual1), " ", get_fitness!(individual2), " new fitness: ", get_fitness!(new_individual1), " ", get_fitness!(new_individual2))
    end

    ga._population = new_population
    _check_new_best!(ga)

    println("Best fitness: ", get_fitness!(ga.best_individual))
end

# function FIHC!(individual::Individual)
#     print("\npre FIHC fitness: ", get_fitness!(individual))

#     random_order = Random.randperm(length(individual.genes))
#     actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)
#     for gene_index in random_order
#         previous_fitness = get_fitness!(individual)
#         previous_gene = individual.genes[gene_index]
#         random_loci_order = [i for i in Random.randperm(actions_number) if i != previous_gene]
#         for loci in random_loci_order
#             individual.genes[gene_index] = loci
#             individual._fitness_actual = false
#             new_fitness = get_fitness!(individual)
#             if new_fitness > previous_fitness
#                 break
#             else
#                 individual.genes[gene_index] = previous_gene
#                 individual._fitness = previous_fitness
#             end
#         end
#     end

#     print("\tpost FIHC fitness: $(get_fitness!(individual))\n")
# end



function FIHC!(individual::Individual)
    # print("\npre FIHC fitness: ", get_fitness!(individual))
    root = individual.env_wrapper._similarity_tree
    tree_levels = [[root.left, root.right]]
    actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)
    individual.genes .= Random.rand(1:actions_number)
    individual._fitness_actual = false
    println(" FIHC initial fitness: ", get_fitness!(individual))

    # this one is test only!!!
    i = 1
    while i <= length(tree_levels) && i <= 10 # 7
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

            #     if old_fitness >= new_fitness
            #         individual.genes[node.elements] = old_elements
            #         individual._fitness = old_fitness
            #     else
            #         println("improvement from $old_fitness to $new_fitness\ttree level $i")
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

            max_copy = argmax(get_fitness!, individuals_copies)
            if get_fitness!(max_copy) > old_fitness
                individual.genes[node.elements] = max_copy.genes[node.elements]
                individual._fitness = max_copy._fitness
                # println("improvement from $old_fitness to $(get_fitness!(max_copy))\ttree level $i")
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

    # print("\tpost FIHC fitness: $(get_fitness!(individual))\n")
end

function get_best_genes(ga::GA_Struct) :: Vector{Int}
    return ga.best_individual.genes
end

function get_best_n_genes(ga::GA_Struct, n::Int) :: Vector{Vector{Int}}
    return [individual.genes for individual in sort(ga._population, by=get_fitness!, rev=true)[1:n]]
end

function get_all_genes(ga::GA_Struct) :: Vector{Vector{Int}}
    return [individual.genes for individual in ga._population]
end

function actualize_population!(ga::GA_Struct, changed_solutions::Vector{Vector{Int}})
    println("population_length: ", length(ga._population), " changed_solutions_length: ", length(changed_solutions))
    for (individual, changed_solution) in zip(ga._population, changed_solutions)
        actualize_genes!(individual, changed_solution)
    end

    ga.best_individual = sort(ga._population, by=get_fitness!, rev=true)[1]
end

function actualize_genes!(individual::Individual, new_genes::Vector{Int})
    println("Actualizing genes, previous genes length: ", length(individual.genes), " new genes length: ", length(new_genes))
    individual.genes = new_genes
    individual._fitness_actual = false

    println("previous fitness: ", individual._fitness, " new fitness: ", get_fitness!(individual))
end

end # module NormalGA