module Environment
    include("../neural_network/NeuralNetwork.jl")
    import .NeuralNetwork
    export AbstractEnvironment, get_safe_data, load_safe_data!, reset!, react!, get_state, is_alive, get_trajectory_data!, get_trajectory_rewards!
    abstract type AbstractEnvironment end

    struct Trajectory
        states::Array{Float32}
        actions::Array{Float32}
        rewards::Array{Float64}
        rewards_sum::Float64
    end

    function Trajectory(states::Array{Float32}, actions::Array{Float32}, rewards::Array{Float64})
        rewards_sum = sum(rewards)
        return Trajectory(states, actions, rewards, rewards_sum)
    end

    # ------------------------------------------------------------------------------------------------
    # Interface functions

    "resets the environment before, doesnt reset after, concrete implementation will have some kwargs"
    function visulize!(env::AbstractEnvironment, model::NeuralNetwork.AbstractNeuralNetwork;)
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
        envs_alive = Vector((env, i) for (i, env) in enumerate(envs))

        if reset
            for env in envs
                reset!(env)
            end
        end

        while length(envs_alive) > 0
            states = Array([get_state(env) for (env, _) in envs_alive])
            actions = NeuralNetwork.predict(neural_network, states)
            i = 1
            while i <= length(envs_alive)
                (env, j) = envs_alive[i]
                rewards[j] += react!(env, actions[i])
                if !is_alive(env)
                    deleteat!(envs_alive, i)
                    deleteat!(actions, i)
                    i -= 1
                end
                i += 1
            end
        end

        return rewards
    end

    "Get the rewards, states and actions of the trajectory of the environments using the neural network. Returns sum of rewards for each environment. Modifies state of environments - resets them before and leaves them used"
    function get_trajectory_data!(envs::Vector{<:AbstractEnvironment}, neural_network::NeuralNetwork.AbstractNeuralNetwork, reset::Bool = true) :: Vector{Trajectory}
        envs_alive = Vector((env, i) for (i, env) in enumerate(envs))
        trajectory_data = Vector{Tuple{Vector{AbstractFloat}, Array{AbstractFloat}, Array{AbstractFloat}}}()

        for env in envs
            if reset
                reset!(env)
            end

            push!(trajectory_data, (Vector{Float64}, Array{Float32}(), Array{Float32}()))
        end

        while length(envs_alive) > 0
            states = Array([get_state(env) for (env, _) in envs_alive])
            actions = NeuralNetwork.predict(neural_network, states)
            i = 1
            while i <= length(envs_alive)
                (env, j) = envs_alive[i]
                reward = react!(env, actions[i])
                push!(trajectory_data[j][1], reward)
                push!(trajectory_data[j][2], states[i])
                push!(trajectory_data[j][3], actions[i])

                if !is_alive(env)
                    deleteat!(envs_alive, i)
                    deleteat!(actions, i)
                    deleteat!(states, i)
                    i -= 1
                end
                i += 1
            end
        end

        return [Trajectory(states, actions, rewards) for (rewards, states, actions) in trajectory_data]
    end

end # module
