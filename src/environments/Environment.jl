
module Environment
    import ..NeuralNetwork

    export AbstractEnvironment, get_safe_data, load_safe_data!, reset!, react!, get_state, get_state_size, get_action_size, is_alive, get_trajectory_data!, get_trajectory_rewards!, get_environment, prepare_environments_kwargs, visualize!
    abstract type AbstractEnvironment end

    struct Trajectory{A1 <: Array{Float32}, A2 <: Array{Float32}}
        states::A1
        actions::A2
        rewards::Vector{Float64}
        rewards_sum::Float64
    end

    function Trajectory(states::Array{Float32}, actions::Array{Float32}, rewards::Array{Float64})
        rewards_sum = sum(rewards)
        return Trajectory(states, actions, rewards, rewards_sum)
    end

    # ------------------------------------------------------------------------------------------------
    # Interface functions

    "Doesnt reset environment afterwards, real implementation will have some kwargs"
    function visualize!(env::AbstractEnvironment, model::NeuralNetwork.AbstractNeuralNetwork, reset::Bool = true;)
        throw("unimplemented")
    end

    function get_state_size(env::AbstractEnvironment)::Vector{Int}
        throw("unimplemented")
    end

    function get_action_size(env::AbstractEnvironment)::Int
        throw("unimplemented")
    end

    function get_safe_data(env::AbstractEnvironment)::Dict{Symbol}
        throw("unimplemented")
    end

    function load_safe_data!(env::AbstractEnvironment, data::Dict{Symbol}) 
        throw("unimplemented")
    end

    function reset!(env::AbstractEnvironment)
        throw("unimplemented")
    end

    function react!(env::AbstractEnvironment, actions::Vector{Float32}) :: Float64
        throw("unimplemented")
    end

    function get_state(env::AbstractEnvironment) :: Vector{Float32}
        throw("unimplemented")
    end

    function is_alive(env::AbstractEnvironment)::Bool
        throw("unimplemented")
    end

    # ------------------------------------------------------------------------------------------------

    # Some general functions, not interface functions

    "Get the rewards of the trajectory of the environments using the neural network. Returns sum of rewards for each environment. Modifies state of environments - resets them before and leaves them used"
    function get_trajectory_rewards!(envs::Vector{<:AbstractEnvironment}, neural_network::NeuralNetwork.AbstractNeuralNetwork; reset::Bool = true) :: Vector{Float64}
        rewards = zeros(Float64, length(envs))
        envs_alive = [(env, i) for (i, env) in enumerate(envs)]

        if reset
            for env in envs
                reset!(env)
            end
        end

        while length(envs_alive) > 0
            states = Array(hcat([get_state(env) for (env, _) in envs_alive]...))
            actions = NeuralNetwork.predict(neural_network, states)
            i = 1
            while i <= length(envs_alive)
                (env, j) = envs_alive[i]
                rewards[j] += react!(env, actions[:, i])
                if !is_alive(env)
                    deleteat!(envs_alive, i)
                    actions = hcat(actions[:, 1:i-1], actions[:, i+1:end])
                    i -= 1
                end
                i += 1
            end
        end

        return rewards
    end

    "Get the rewards, states and actions of the trajectory of the environments using the neural network. Returns sum of rewards for each environment. Modifies state of environments - resets them before and leaves them used"
    function get_trajectory_data!(envs::Vector{<:AbstractEnvironment}, neural_network::NeuralNetwork.AbstractNeuralNetwork, reset::Bool = true) :: Vector{Trajectory}
        envs_alive = [(env, i) for (i, env) in enumerate(envs)]
        trajectory_data = Vector{Tuple{Vector{Float64}, Vector{Array{Float32}}, Vector{Array{Float32}}}}()

        for env in envs
            if reset
                reset!(env)
            end

            push!(trajectory_data, (Vector{Float64}(), Vector{Array{Float32}}(), Vector{Array{Float32}}()))
        end

        while length(envs_alive) > 0
            states = reduce(hcat, [get_state(env) for (env, _) in envs_alive])
            actions = NeuralNetwork.predict(neural_network, states)
            i = 1
            while i <= length(envs_alive)
                (env, j) = envs_alive[i]
                current_action = actions[:, i]
                reward = react!(env, current_action)
                push!(trajectory_data[j][1], reward)
                push!(trajectory_data[j][2], states[:, i])
                push!(trajectory_data[j][3], current_action)

                if !is_alive(env)
                    deleteat!(envs_alive, i)
                    states = hcat(states[:, 1:i-1], states[:, i+1:end])
                    actions = hcat(actions[:, 1:i-1], actions[:, i+1:end])
                    i -= 1
                end
                i += 1
            end
        end

        return [Trajectory(reduce(hcat, states), reduce(hcat, actions), rewards) for (rewards, states, actions) in trajectory_data]
    end



    # includes
    include("_CarEnvironment.jl")
    using .CarEnvironment


    function get_environment(name::Symbol) :: Type
        if name == :BasicCarEnvironment
            return BasicCarEnvironment
        else
            throw("Environment not found")
        end
    end

    function prepare_environments_kwargs(dict_universal::Dict{Symbol, Any}, dict_changeable::Vector{Dict{Symbol, Any}}) :: Vector{Dict{Symbol, Any}}
        dicts_copy = [deepcopy(dict_universal) for _ in 1:length(dict_changeable)]
        for i in 1:length(dict_changeable)
            for (key, value) in dict_changeable[i]
                dicts_copy[i][key] = value
            end
        end

        return dicts_copy
    end
end # module
