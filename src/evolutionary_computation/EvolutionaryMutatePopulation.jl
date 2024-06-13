module EvolutionaryMutatePopulaiton
    include("../neural_network/NeuralNetwork.jl")
    include("../neural_network/NeuralNetworkFunctions.jl")
    include("../environments/EnvironmentFunctions.jl")
    include("../environments/Environment.jl")

    using Statistics
    using .NeuralNetwork
    using .NeuralNetworkFunctions
    using .EnvironmentFunctions
    using .Environment

    export EvolutionaryMutatePopulationAlgorithm, run!

    #--------------------------------------------------------------------------------------------------------
    # protected

    mutable struct Individual{T <: Environment.AbstractEnvironment, N <: NeuralNetwork.AbstractNeuralNetwork}
        const neural_network_type::Type{N}
        const neural_network::N
        const neural_network_kwargs::Dict{Symbol, Any}
        _fitness::Float64
        _is_fitness_calculated::Bool
        const environments::Vector{T}
        const environments_kwargs::Vector{Dict{Symbol, Any}}
        const environment_type::Type{T}
    end

    function Individual(
            neural_network_type::Type{T},
            neural_network_kwargs::Dict{Symbol, Any},
            environments_kwargs::Dict{Symbol, Any},
            environment_type::Type{N}
        ) #where T <: Environment.AbstractEnvironment where N <: NeuralNetwork.AbstractNeuralNetwork
        return Individual{environment_type, neural_network_type}(
            neural_network_type,
            (neural_network_type)(neural_network_kwargs...),
            neural_network_kwargs,
            0.0,
            false,
            [(environment_type)(;environment_kwarg...) for environment_kwarg in environments_kwargs],
            environments_kwargs,
            environment_type
        )
    end

    function get_fitness(ind::Individual)::Float64
        if !ind._is_fitness_calculated
            ind._fitness = sum(get_trajectory_rewards!(ind.environments, ind.neural_network; reset=true))
            ind._is_fitness_calculated = true
        end
        return ind._fitness
    end

    "Returns new individual with mutated neural network and the same environments."
    function mutate(ind::Individual, mutation_rate::Float64) :: Individual
        get_fitness(ind)
        new_individual = copy(ind)
        params = get_parameters(new_individual.neural_network)
        
        for param in params
            param .+= randn() * mutation_rate
        end

        set_parameters!(new_individual.neural_network, params)
        new_individual._is_fitness_calculated = false
        get_fitness(new_individual)
        return new_individual
    end

    function copy(ind::Individual) :: Individual
        new_individual = Individual(
            ind.neural_network_type,
            ind.neural_network_kwargs,
            ind.environments_kwargs,
            ind.environment_type
        )
        set_parameters!(new_individual.neural_network, get_parameters(ind.neural_network))
        new_individual._fitness = ind._fitness
        new_individual._is_fitness_calculated = ind._is_fitness_calculated
        return new_individual
    end


    #--------------------------------------------------------------------------------------------------------
    # public

    mutable struct EvolutionaryMutatePopulationAlgorithm
        population::Vector{Individual}
        population_size::Int
        mutation_rate::Float64
        const max_generations::Int
        const max_evaluations::Int
        const visualization_kwargs::Dict{Symbol, Any}
        const visualization_environment::Environment.AbstractEnvironment
        best_individual::Individual
        const n_threads::Int
    end

    function EvolutionaryMutatePopulationAlgorithm(;
        population_size::Int,
        mutation_rate::Float64,
        max_generations::Int,
        max_evaluations::Int,
        n_threads::Int,
        environment_kwargs::Vector{Dict{Symbol, Any}},
        visualization_kwargs::Dict{Symbol, Any},
        environment_visualization_kwargs::Dict{Symbol, Any},
        environment::Symbol,
        neural_network_data::Dict{Symbol, Any}
    ) :: EvolutionaryMutatePopulationAlgorithm
        nn_type = get_neural_network(neural_network_data[:name])
        env_type = get_environment(environment)
        visualization_environment = (env_type)(;environment_visualization_kwargs...)

        println((nn_type, env_type), (typeof(nn_type), typeof(env_type)))

        ind = Individual(
            AbstractNeuralNetwork,
            neural_network_data[:kwargs],
            environment_kwargs,
            AbstractEnvironment
        )

        println("Hello world")

        population = [
            Individual(
                nn_type,
                neural_network_data[:kwargs],
                environment_kwargs,
                env_type
        ) for _ in 1:population_size]

        visualization_environment = (env_type)(;environment_visualization_kwargs...)

        return EvolutionaryMutatePopulationAlgorithm(
            population,
            population_size,
            mutation_rate,
            max_generations,
            max_evaluations,
            visualization_kwargs,
            visualization_environment,
            population[1],
            n_threads > 0 ? n_threads : Threads.nthreads()
        )
    end

    function run!(algo::EvolutionaryMutatePopulationAlgorithm)
        for generation in 1:algo.max_generations
            for individual in algo.population
                if generation * algo.population_size >= algo.max_evaluations
                    break
                end

                # new_individuals = Threads.foreach((fit_individual) -> mutate(fit_individual, algo.mutation_rate), algo.population; ntasks=Threads.threadpoolsize())
                time_start = time() 
                new_individuals = Threads.@threads for i in 1:length(algo.population)
                    mutate(algo.population[i], algo.mutation_rate)
                end
                time_end = time()
                println("Generation: $generation, time: $(time_end - time_start)")
                
                append!(algo.population, new_individuals)
                sorted = sort(algo.population, by=get_fitness, rev=true)
                algo.population = sorted[1:algo.population_size]

                if get_fitness(algo.population[1]) > get_fitness(algo.best_individual)
                    algo.best_individual = copy(algo.population[1])
                end

                println("best: $(get_fitness(algo.best_individual))    mean: $(mean(get_fitness.(algo.population)))\n\n\n")
            end
        end
    end
end