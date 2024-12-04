module EvolutionaryMutatePopulaiton
    
import ..NeuralNetwork
import ..Environment

import Statistics as St
import Printf as Pf
import JLD
import Random

# tmp
import Flux
import Plots

export EvolutionaryMutatePopulationAlgorithm, run!

#--------------------------------------------------------------------------------------------------------
# protected

mutable struct Individual{T<:Environment.AbstractEnvironment, N<:NeuralNetwork.AbstractNeuralNetwork}
    const neural_network_type::Type{N}
    const neural_network::N
    const neural_network_kwargs::Dict{Symbol, Any}
    # const parent::Union{Individual{T,N}, Nothing}
    _fitness::Float64
    _trajectory::Vector{Environment.Trajectory}
    _is_fitness_calculated::Bool
    const environments::Vector{T}
    const environments_kwargs::Vector{Dict{Symbol, Any}}
    const environment_type::Type{T}
end

function Individual(
        neural_network_type::Type{N},
        neural_network_kwargs::Dict{Symbol, Any},
        environments_kwargs::Vector{Dict{Symbol, Any}},
        environment_type::Type{T},
        # parent=nothing
    ) where {T<:Environment.AbstractEnvironment, N<:NeuralNetwork.AbstractNeuralNetwork}
    return Individual{environment_type, neural_network_type}(
        neural_network_type,
        (neural_network_type)(;neural_network_kwargs...),
        neural_network_kwargs,
        # parent,
        0.0,
        Vector{Environment.Trajectory}(undef, 0),
        false,
        [(environment_type)(;environment_kwarg...) for environment_kwarg in environments_kwargs],
        environments_kwargs,
        environment_type
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

function local_search(ind::Individual) :: Individual
    TRIES = 20
    CHANGES = 5
    
    previous_fitness = get_fitness(ind)
    new_individual = copy(ind)
    trajectories = Environment.get_trajectory_data!(new_individual.environments, new_individual.neural_network)
    
    for _ in 1:TRIES
        states = reduce(hcat, [trajectory.states for trajectory in trajectories])
        JLD.save("log/states.jld", "states", states)
        actions = reduce(hcat, [trajectory.actions for trajectory in trajectories])

        random_index = rand(1:size(states)[2], CHANGES)
        random_states = states[:, random_index]
        

        random_actions = actions[:, random_index]

        random_actions .+= randn(Float32, size(random_actions)) .* 0.1
        random_actions = vcat(Flux.softmax(random_actions[1:3, :]), Flux.softmax(random_actions[4:6, :])) #(random_actions .- minimum(random_actions, dims=1)) ./ (maximum(random_actions, dims=1) .- minimum(random_actions, dims=1))
        
        previous_params = NeuralNetwork.get_parameters(new_individual.neural_network)
        NeuralNetwork.learn!(new_individual.neural_network, random_states, random_actions, Flux.kldivergence; epochs=10, learning_rate=0.001)
        
        trajectories_new = Environment.get_trajectory_data!(new_individual.environments, new_individual.neural_network)
        new_fitness = sum([trajectory.rewards_sum for trajectory in trajectories_new])
        if new_fitness > previous_fitness
            previous_fitness = new_fitness
            trajectories = trajectories_new
        else
            NeuralNetwork.set_parameters!(new_individual.neural_network, previous_params)
        end
    end

    new_individual._fitness = previous_fitness
    new_individual._is_fitness_calculated = true
    return new_individual
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

function differential_evolution_one_run(base_ind::Individual, ind_a::Individual, ind_b::Individual)::Individual
    new_ind = copy(base_ind)
    states = reduce(hcat, [trajectory.states for trajectory in get_trajectories(new_ind)])
    actions_new_ind = reduce(hcat, [trajectory.actions for trajectory in get_trajectories(new_ind)])

    actions_a = NeuralNetwork.predict(ind_a.neural_network, states)
    actions_b = NeuralNetwork.predict(ind_b.neural_network, states)

    a_b_normalized = actions_a - actions_b
    # a_b_normalized .-= minimum(a_b_normalized, dims=1)
    # a_b_normalized ./= sum(a_b_normalized, dims=1)

    actions_new_ind += a_b_normalized
    actions_new_ind .-= minimum(actions_new_ind, dims=1)
    actions_new_ind ./= sum(actions_new_ind, dims=1)

    NeuralNetwork.learn!(new_ind.neural_network, states, actions_new_ind; epochs=3, learning_rate=0.001, verbose=false)

    new_ind._is_fitness_calculated = false
    get_fitness(new_ind)

    return new_ind
end

function copy(ind::Individual) :: Individual
    new_individual = Individual(
        ind.neural_network_type,
        ind.neural_network_kwargs,
        ind.environments_kwargs,
        ind.environment_type,
        # ind
    )
    NeuralNetwork.set_parameters!(new_individual.neural_network, NeuralNetwork.get_parameters(ind.neural_network))
    new_individual._fitness = ind._fitness
    new_individual._is_fitness_calculated = ind._is_fitness_calculated
    new_individual._trajectory = ind._trajectory
    return new_individual
end


#--------------------------------------------------------------------------------------------------------
# public

mutable struct EvolutionaryMutatePopulationAlgorithm
    population::Vector{Individual{T, N}} where {T<:Environment.AbstractEnvironment, N<:NeuralNetwork.AbstractNeuralNetwork}
    population_size::Int
    mutation_rate::Float64
    const visualization_kwargs::Dict{Symbol, Any}
    const visualization_environment::Environment.AbstractEnvironment
    best_individual::Individual
    const n_threads::Int
end

function EvolutionaryMutatePopulationAlgorithm(;
    population_size::Int,
    mutation_rate::Float64,
    n_threads::Int,
    environment_kwargs::Vector{Dict{Symbol, Any}},
    visualization_kwargs::Dict{Symbol, Any},
    environment_visualization_kwargs::Dict{Symbol, Any},
    environment::Symbol,
    neural_network_data::Dict{Symbol, Any}
) :: EvolutionaryMutatePopulationAlgorithm
    nn_type = NeuralNetwork.get_neural_network(neural_network_data[:name])
    env_type = Environment.get_environment(environment)
    
    visualization_environment = (env_type)(;environment_visualization_kwargs...)

    population = Vector([
        Individual(
            nn_type,
            neural_network_data[:kwargs],
            environment_kwargs,
            env_type
    ) for _ in 1:population_size])

    visualization_environment = (env_type)(;environment_visualization_kwargs...)

    return EvolutionaryMutatePopulationAlgorithm(
        population,
        population_size,
        mutation_rate,
        visualization_kwargs,
        visualization_environment,
        copy(population[1]),
        n_threads > 0 ? n_threads : Threads.nthreads()
    )
end

function run!(algo::EvolutionaryMutatePopulationAlgorithm; max_generations::Int, max_evaluations::Int, log::Bool=true, visualize_each_n_epochs::Int=0)
    for generation in 1:max_generations
        if generation * algo.population_size >= max_evaluations
            break
        end

        time_start = time()

        new_individuals = Vector(algo.population)
        Threads.@threads for i in eachindex(new_individuals)
        # for i in eachindex(new_individuals)
            # new_individuals[i] = @time mutate(new_individuals[i], algo.mutation_rate)
            new_individuals[i] = mutate(new_individuals[i], algo.mutation_rate)
        end
        append!(algo.population, new_individuals)

        # rand_permutation = Random.randperm(length(algo.population))
        # Threads.@threads for i in 1:2:99 # 2:10  # Threads.@threads
        #     parent1 = algo.population[rand_permutation[i]]
        #     parent2 = algo.population[rand_permutation[i+1]]
        #     new_individual = crossover(parent1, parent2)
        #     index_to_replace = get_fitness(parent1) < get_fitness(parent2) ? i : i + 1
        #     algo.population[rand_permutation[index_to_replace]] = new_individual
        # end

        append!(algo.population, new_individuals)

        # differential evolution test
        rand_permutation = Random.randperm(length(algo.population))
        Threads.@threads for i in 1:2:99 # 2:10  # Threads.@threads
            parent1 = algo.population[rand_permutation[i]]
            parent2 = algo.population[rand_permutation[i+1]]
            new_individual = differential_evolution_one_run(algo.best_individual, parent1, parent2)
            if get_fitness(new_individual) > get_fitness(parent1)
                algo.population[rand_permutation[i]] = new_individual
            end
        end

        sorted = sort(algo.population, by=get_fitness, rev=true)
        algo.population = sorted[1:algo.population_size]
        time_end = time()

        if get_fitness(algo.population[1]) > get_fitness(algo.best_individual)
            # locally_new = local_search(algo.best_individual)

            # if log
            #     println("\n\n\n\n$(generation) - new best fitness:")
            #     Pf.@printf "previous: %.2f \t new: %.2f \t new ls: %.2f \n\n\n\n" get_fitness(algo.best_individual) get_fitness(algo.population[1]) get_fitness(locally_new)
            # end
            # algo.population[1] = locally_new
            algo.best_individual = copy(algo.population[1])

            if get_fitness(algo.best_individual) > 400.0
                # _save_previous_responses(algo)
            end
        end

        if log
            quantiles = [0.25, 0.5, 0.75, 0.95]
            quantiles_values = St.quantile(get_fitness.(algo.population), quantiles)
            elapsed_time = time_end - time_start
            one_mutation_ratio = Threads.nthreads() * elapsed_time / length(algo.population)
            Pf.@printf "Generation: %i, time: %.3f, threads: %i, calculated: %i, time*threads/calc: %.3f\n" generation elapsed_time Threads.nthreads() length(algo.population_size) one_mutation_ratio
            Pf.@printf "best: %.2f\tmean: %.2f\n" get_fitness(algo.best_individual) St.mean(get_fitness.(algo.population))
            println("quantiles:\t$(join([(Pf.@sprintf "%.2f: %.2f" quantile fitness) for (quantile, fitness) in zip(quantiles, quantiles_values)], "\t"))")
            print("\n\n\n")
        end

        if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
            Environment.visualize!(algo.visualization_environment, algo.best_individual.neural_network; algo.visualization_kwargs...)
        end
    end
end

function _save_previous_responses(algo::EvolutionaryMutatePopulationAlgorithm, save_dir::String = "log/_evolutionary_mutate_population/")
    trajectories = Environment.get_trajectory_data!(algo.best_individual.environments, algo.best_individual.neural_network)
    trajectory = trajectories[1]
    # states = [trajectory.states[:, i] for i in 1:size(trajectory.states)[2]]
    states = hcat([trajectory.states for trajectory in trajectories]...)
    actions = hcat([trajectory.actions for trajectory in trajectories]...)
    JLD.save("$(save_dir)states.jld", "states", states)
    actions_vector = Vector{Union{String, Vector{Int}}}()
    individual = algo.best_individual
    individual_id = 0
    while !isnothing(individual)
        actions = NeuralNetwork.predict(individual.neural_network, states)
        actions_ids = [argmax(actions[:, i]) for i in 1:size(actions, 2)]
        push!(actions_vector, "actions_$(individual_id)_f$(individual._fitness).jld", actions_ids)

        individual = individual.parent
        individual_id += 1
    end
    JLD.save("$(save_dir)actions.jld", actions_vector...)

    display(states)
    # states_cosine_similarity = [_cosine_similarity(states[1], state) for state in states]
    # decisions_ids = [Vector{Int}() for _ in 1:9]
    # for i in 1:size(actions, 2)
    #     push!(decisions_ids[argmax(actions[:, i])], i)
    # end

    # # save states
    # for i in 1:9
    #     JLD.save("log/states_$i.jld", "states", states[:, decisions_ids[i]])
    # end
end

function _cosine_similarity(v1::Vector{Float32}, v2::Vector{Float32}) :: Float64
    value = 0.0
    length_1 = 0.0
    length_2 = 0.0

    for (x, y) in zip(v1, v2)
        value += x * y
        length_1 += x^2
        length_2 += y^2
    end

    return value / (sqrt(length_1) * sqrt(length_2))
end 
end