
# it has different state of environment on each level (next levels are more actualised)
# therefore in general I do not have to actualise fitnesses of individuals - only the one that moves from one level to another, so I can FIHC it
module P3Levels

import ..NeuralNetwork
import ..Environment
import ..EnvironmentWrapper

import Clustering
import Random
import Plots
import Dates

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct Individual
    genes::Vector{Int}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
    _fitness::Float64
    _fitness_actual::Bool
end

struct Level 
    individuals::Vector{Individual}
    env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct
end

mutable struct Population_Pyramid
    population::Vector{Level}
    # _dependencies_matrices::Vector{Array{Int, 4}}
    best_individual::Individual
end

# --------------------------------------------------------------------------------------------------
# Public functions

function Individual(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    genes = Random.rand(1:EnvironmentWrapper.get_action_size(env_wrapper), EnvironmentWrapper.get_groups_number(env_wrapper))
    fitness = -Inf
    fitness_actual = false
    return Individual(genes, env_wrapper, fitness, fitness_actual)
end

function Population_Pyramid(env_wrapper::EnvironmentWrapper.EnvironmentWrapperStruct)
    first_individual = Individual(env_wrapper)
    @time FIHC!(first_individual)
    return Population_Pyramid([Level([first_individual], env_wrapper)], first_individual)
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

function actualize_genes!(individual::Individual, new_genes::Vector{Int})
    individual.genes = new_genes
    individual._fitness_actual = false
end

function optimal_mixing!(individual::Individual, other_individuals::Vector{Individual})
    print("\npre optimal mixing fitness: ", get_fitness!(individual))
    root = individual.env_wrapper._similarity_tree
    tree_levels = [[root.left, root.right]]

    # this one is test only!!!
    i = 1
    while i <= length(tree_levels) && i <= 7
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
                individual_tmp = copy_individual(individual)
                individuals_copies[i] = individual_tmp
                if individual.genes[node.elements] != other_individuals[i].genes[node.elements]
                    individual_tmp.genes[node.elements] = other_individuals[i].genes[node.elements]
                    individual_tmp._fitness_actual = false
                    get_fitness!(individual_tmp)
                end
            end

            max_copy = argmax(get_fitness!, individuals_copies)
            if get_fitness!(max_copy) > old_fitness
                individual.genes[node.elements] = max_copy.genes[node.elements]
                individual._fitness = max_copy._fitness
                println("improvement from $old_fitness  to $(get_fitness!(max_copy))\ttree level $i")
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

    print("\tpost optimal mixing fitness: $(get_fitness!(individual))\n")
end

function FIHC!_off(individual::Individual)
    # flat version
    print("\npre FIHC fitness: ", get_fitness!(individual))
    random_perm = Random.randperm(length(individual.genes))
    actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)

    for gene_index in random_perm
        previous_fitness = get_fitness!(individual)
        previous_gene = individual.genes[gene_index]
        random_loci_order = [i for i in Random.randperm(actions_number) if i != previous_gene]
        for loci in random_loci_order
            individual.genes[gene_index] = loci
            individual._fitness_actual = false
            new_fitness = get_fitness!(individual)
            if new_fitness > previous_fitness
                print("improvement from $previous_fitness to $new_fitness\tgene index $gene_index\n")
            else
                individual.genes[gene_index] = previous_gene
                individual._fitness = previous_fitness
            end
        end
    end
    print("\tpost FIHC fitness: $(get_fitness!(individual))\n")
end

function FIHC!(individual::Individual)
    println("\npre FIHC fitness: $(get_fitness!(individual))\n")
    root = individual.env_wrapper._similarity_tree
    tree_levels = [[root.left, root.right]]
    actions_number = EnvironmentWrapper.get_action_size(individual.env_wrapper)
    individual.genes .= Random.rand(1:actions_number)
    individual._fitness_actual = false
    println(" FIHC initial fitness: ", get_fitness!(individual))
    overall_fitness_ckecks = 0

    # this one is test only!!!
    i = 1
    while i <= length(tree_levels) && i <= 9
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

            overall_fitness_ckecks += actions_number

            max_copy = argmax(get_fitness!, individuals_copies)
            if get_fitness!(max_copy) > old_fitness
                individual.genes[node.elements] = max_copy.genes[node.elements]
                individual._fitness = max_copy._fitness
                println("improvement from $old_fitness  to $(get_fitness!(max_copy))\ttree level $i")
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

    print("overall_fitness_ckecks: $overall_fitness_ckecks")

    print("\tpost FIHC fitness: $(get_fitness!(individual))\n")
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
    Plots.savefig("log/_P3_FIHC_output/$(timestamp_string)_$(get_fitness!(individual)).png")
end

"RUns new individual, does FIHC, optimal mixing, climbing through levels and returns final individual"
function run_new_individual!(p3::Population_Pyramid)
    new_individual = Individual(p3.population[1].env_wrapper)
    @time FIHC!(new_individual)
    # save_decision_plot(new_individual)
    add_to_next_level = true

    i = 1
    while i <= length(p3.population)
        new_individual = copy_individual(new_individual)
        if i > 1
            new_env_wrapper = p3.population[i].env_wrapper
            new_individual.genes = EnvironmentWrapper.translate_solutions(new_individual.env_wrapper, new_env_wrapper, [new_individual.genes])[1]
            new_individual.env_wrapper = new_env_wrapper
            old_fitness = get_fitness!(new_individual)
            new_individual._fitness_actual = false
            println("fitness changed from $old_fitness to $(get_fitness!(new_individual))")
            @time FIHC!(new_individual)
        end

        if add_to_next_level
            push!(p3.population[i].individuals, new_individual)
        end

        old_fitness = get_fitness!(new_individual)
        optimal_mixing!(new_individual, p3.population[i].individuals)
        new_fitness = get_fitness!(new_individual)

        if new_fitness > old_fitness
            add_to_next_level = true

            if i == length(p3.population)
                new_env_wrapper = EnvironmentWrapper.copy(p3.population[i].env_wrapper)
                # actually I do not want to actualise any genes, i just want to create new empty level
                genes = [individual.genes for individual in p3.population[i].individuals]
                EnvironmentWrapper.actualize!(new_env_wrapper, genes, genes)
                push!(p3.population, Level(Vector{Individual}(undef, 0), new_env_wrapper))
            end
        else
            add_to_next_level = false
        end

        i += 1
    end
end

end # module P3