mutable struct Individual{T<:Environment.AbstractEnvironment, N<:NeuralNetwork.AbstractNeuralNetwork}
    const neural_network_type::Type{N}
    const neural_network::N
    const neural_network_kwargs::Dict{Symbol, Any}
    _fitness::Float64
    _trajectory::Vector{Environment.Trajectory}
    _is_fitness_calculated::Bool
    const environments::Vector{T}
end

function Individual(
        neural_network_type::Type{N},
        neural_network_kwargs::Dict{Symbol, Any},
        environments::Vector{T},
    ) where {T<:Environment.AbstractEnvironment, N<:NeuralNetwork.AbstractNeuralNetwork}
    return Individual(
        neural_network_type,
        (neural_network_type)(;neural_network_kwargs...),
        neural_network_kwargs,
        -Inf,
        Vector{Environment.Trajectory}(undef, 0),
        false,
        [Environment.copy(environment) for environment in environments]
    )
end

function get_fitness(ind::Individual)::Float64
    if !ind._is_fitness_calculated
        # ind._fitness = sum(Environment.get_trajectory_rewards!(ind.environments, ind.neural_network; reset=true))
        ind._trajectory = Environment.get_trajectory_data!(ind.environments, ind.neural_network)
        ind._fitness = sum([trajectory.rewards_sum for trajectory in ind._trajectory])
        ind._is_fitness_calculated = true
    end
    return ind._fitness
end

function get_trajectories(ind::Individual)
    if !ind._is_fitness_calculated
        get_fitness(ind)
    end
    return ind._trajectory
end


"Returns new individual with mutated neural network and the same environments."
function mutate(ind::Individual, mutation_rate::Float64) :: Individual
    get_fitness(ind)
    new_individual = copy(ind)
    params = NeuralNetwork.get_parameters(new_individual.neural_network)
    
    for param in params
        param .+= randn(Float32, size(param)) .* mutation_rate
    end

    NeuralNetwork.set_parameters!(new_individual.neural_network, params)
    new_individual._is_fitness_calculated = false
    get_fitness(new_individual)
    return new_individual
end

function local_search!(ind::Individual)
    # prepare data
    nn_representant = NeuralNetwork.get_input_representant_nn(ind.neural_network)
    states = reduce(hcat, [trajectory.states for trajectory in get_trajectories(ind)])
    actions = reduce(hcat, [trajectory.actions for trajectory in get_trajectories(ind)])
    encoded_states = NeuralNetwork.predict(nn_representant, states)
    
    tree = create_tree(encoded_states)

    # actual FIHC
    println("\npre FIHC fitness: $(get_fitness(ind))     states number: $(size(states, 2))\n")
    tree_levels = [[tree]]
    actions_number = Environment.get_action_size(ind.environments[1])

    i = 1
    while i <= length(tree_levels) && i <= 8
        current_level = tree_levels[i]
        random_perm = Random.randperm(length(current_level))

        for node in current_level[random_perm]
            random_perm_actions = Random.randperm(actions_number)
            old_actions = deepcopy(actions)
            old_nn_params = NeuralNetwork.get_parameters(ind.neural_network)
            old_trajectories = get_trajectories(ind)
            old_fitness = get_fitness(ind)
            
            for chosen_action in random_perm_actions
                add_vector = zeros(Float32, actions_number)
                add_vector[chosen_action] = 1.0

                elem_actions = @view actions[:, node.elements]
                actions[:, node.elements] .+= add_vector
                actions[:, node.elements] .-= minimum(elem_actions, dims=1)
                actions[:, node.elements] ./= sum(elem_actions, dims=1)
                # actions[:, node.elements] .= add_vector

                NeuralNetwork.learn!(ind.neural_network, states, actions; epochs=5, learning_rate=0.001, verbose=false)

                ind._is_fitness_calculated = false
                new_fitness = get_fitness(ind)

                if old_fitness >= new_fitness
                    NeuralNetwork.set_parameters!(ind.neural_network, old_nn_params)
                    ind._trajectory = old_trajectories
                    ind._fitness = old_fitness
                    actions = old_actions
                    # println("no improvement from $old_fitness to $new_fitness\ttree level $i")
                else
                    println("improvement from $old_fitness to $new_fitness\ttree level $i")
                    # save_decision_plot(individual)
                    # old_fitness = new_fitness
                    # old_actions = copy(actions)
                    # old_nn_params = NeuralNetwork.get_parameters(ind.neural_network)
                    break
                end
            end

            if !is_leaf(node)
                if i == length(tree_levels)
                    push!(tree_levels, [node.left, node.right])
                else
                    push!(tree_levels[i+1], node.left, node.right)
                end
            end
        end

        i += 1
    end
    print("\tpost FIHC fitness: $(get_fitness(ind))\n")
end

function crossover(ind1::Individual, ind2::Individual) :: Individual
    get_fitness(ind1)
    get_fitness(ind2)
    new_individual = copy(ind1)
    trajectories_ind1 = get_trajectories(ind1)
    states = reduce(hcat, [trajectory.states for trajectory in trajectories_ind1])
    actions_ind1 = reduce(hcat, [trajectory.actions for trajectory in trajectories_ind1])
    actions_ind2 = NeuralNetwork.predict(ind2.neural_network, states)
    actions_to_learn = actions_ind1 + actions_ind2
    actions_to_learn = actions_to_learn .- minimum(actions_to_learn, dims=1)
    actions_to_learn = actions_to_learn ./ sum(actions_to_learn, dims=1)
    NeuralNetwork.learn!(new_individual.neural_network, states, actions_to_learn; epochs=3, learning_rate=0.001, verbose=false)

    new_individual._is_fitness_calculated = false
    get_fitness(new_individual)

    # println("fitness 1: $(get_fitness(ind1)), fitness 2: $(get_fitness(ind2)), new fitness: $(get_fitness(new_individual))")

    return new_individual
end

function copy(ind::Individual) :: Individual
    new_individual = Individual(
        ind.neural_network_type,
        ind.neural_network_kwargs,
        ind.environments
    )
    NeuralNetwork.set_parameters!(new_individual.neural_network, NeuralNetwork.get_parameters(ind.neural_network))
    new_individual._fitness = ind._fitness
    new_individual._is_fitness_calculated = ind._is_fitness_calculated
    new_individual._trajectory = ind._trajectory
    return new_individual
end