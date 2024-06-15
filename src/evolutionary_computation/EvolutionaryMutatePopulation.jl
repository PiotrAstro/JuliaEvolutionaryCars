module EvolutionaryMutatePopulaiton
    import ..NeuralNetwork
    import ..Environment
    import Statistics as St
    import Printf as Pf

    export EvolutionaryMutatePopulationAlgorithm, run!

    #--------------------------------------------------------------------------------------------------------
    # protected

    mutable struct Individual{T<:Environment.AbstractEnvironment, N<:NeuralNetwork.AbstractNeuralNetwork}
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
            neural_network_type::Type{N},
            neural_network_kwargs::Dict{Symbol, Any},
            environments_kwargs::Vector{Dict{Symbol, Any}},
            environment_type::Type{T}
        ) where {T<:Environment.AbstractEnvironment, N<:NeuralNetwork.AbstractNeuralNetwork}
        return Individual{environment_type, neural_network_type}(
            neural_network_type,
            (neural_network_type)(;neural_network_kwargs...),
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
            ind._fitness = sum(Environment.get_trajectory_rewards!(ind.environments, ind.neural_network; reset=true))
            ind._is_fitness_calculated = true
        end
        return ind._fitness
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

    function crossover(ind1::Individual, ind2::Individual) :: Individual
        get_fitness(ind1)
        get_fitness(ind2)
        new_individual = copy(ind1)
        params1 = NeuralNetwork.get_parameters(ind1.neural_network)
        params2 = NeuralNetwork.get_parameters(ind2.neural_network)
        new_params = [rand() < 0.5 ? param1 : param2 for (param1, param2) in zip(params1, params2)]
        NeuralNetwork.set_parameters!(new_individual.neural_network, new_params)
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
        NeuralNetwork.set_parameters!(new_individual.neural_network, NeuralNetwork.get_parameters(ind.neural_network))
        new_individual._fitness = ind._fitness
        new_individual._is_fitness_calculated = ind._is_fitness_calculated
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
            population[1],
            n_threads > 0 ? n_threads : Threads.nthreads()
        )
    end

    function run!(algo::EvolutionaryMutatePopulationAlgorithm; max_generations::Int, max_evaluations::Int, log::Bool=true, visualize_each_n_epochs::Int=0)
        for generation in 1:max_generations
            if generation * algo.population_size >= max_evaluations
                break
            end

            # new_individuals = Threads.foreach((fit_individual) -> mutate(fit_individual, algo.mutation_rate), algo.population; ntasks=Threads.threadpoolsize())
            time_start = time()
            # new_individuals = Threads.foreach((fit_individual) -> mutate(fit_individual, algo.mutation_rate), algo.population; ntasks=Threads.threadpoolsize())
            new_individuals = Vector(algo.population)

            Threads.@threads for i in eachindex(new_individuals)
            # for i in eachindex(new_individuals)
                new_individuals[i] = mutate(new_individuals[i], algo.mutation_rate)
            end
            time_end = time()


            
            append!(algo.population, new_individuals)
            sorted = sort(algo.population, by=get_fitness, rev=true)
            algo.population = sorted[1:algo.population_size]

            if get_fitness(algo.population[1]) > get_fitness(algo.best_individual)
                algo.best_individual = algo.population[1]
            end

            if log
                quantiles = [0.25, 0.5, 0.75, 0.95]
                quantiles_values = St.quantile(get_fitness.(algo.population), quantiles)
                elapsed_time = time_end - time_start
                one_mutation_ratio = Threads.nthreads() * elapsed_time / length(new_individuals)
                Pf.@printf "Generation: %i, time: %.3f, threads: %i, calculated: %i, time*threads/calc: %.3f\n" generation elapsed_time Threads.nthreads() length(new_individuals) one_mutation_ratio
                Pf.@printf "best: %.2f\tmean: %.2f\n" get_fitness(algo.best_individual) St.mean(get_fitness.(algo.population))
                println("quantiles:\t$(join([(Pf.@sprintf "%.2f: %.2f" quantile fitness) for (quantile, fitness) in zip(quantiles, quantiles_values)], "\t"))")
                print("\n\n\n")
            end

            if visualize_each_n_epochs > 0 && generation % visualize_each_n_epochs == 0
                Environment.visualize!(algo.visualization_environment, algo.best_individual.neural_network; algo.visualization_kwargs...)
            end
        end
    end
end