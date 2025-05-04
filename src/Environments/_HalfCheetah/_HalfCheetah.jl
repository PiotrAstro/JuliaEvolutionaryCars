# this is a translation of Farama-Foundation Gymnasium HalfCheetah implementation to pure Julia environment
# their implementation: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/humanoid_v5.py

# this test sometimes passes, sometimes not
# I assume it might be the problem with float precision, this error is accumulating over time
# therefore I think it is not correctness problem, as the same issue will occur between python versions etc.
# Mayb different MuJoCo version? idk.
@testitem "HalfCheetah-v5" begin
    import PythonCall
    import JuliaEvolutionaryCars.Environment as Environment
    import Test
    
    gym = PythonCall.pyimport("gymnasium")
    py_cheetah = gym.make("HalfCheetah-v5", reset_noise_scale=0.0)
    py_cheetah.reset()
    my_cheetah = Environment.HalfCheetah(base_version=:half_cheetah_v5, reset_noise_scale=0.0, max_steps=70)
    my_cheetah.reset!()

    random_nn = Environment.NeuralNetwork.Random_NN(Environment.get_action_size(my_cheetah))
    while Environment.is_alive(my_cheetah) && my_cheetah.current_step < my_cheetah.max_steps -1
        action = Environment.NeuralNetwork.predict(random_nn, Environment.NeuralNetwork.MatrixASSEQ([ones(Float32, 1)]))[:, 1]
        action = clamp.(action, -1.0f0, 1.0f0)
        reward = Environment.react!(my_cheetah, action)
        state = Environment.get_state(my_cheetah)
        py_state, py_reward, done, _, info = py_cheetah.step(action)
        done = PythonCall.pyconvert(Bool, done)
        py_state = PythonCall.pyconvert(Vector{Float32}, py_state)
        py_reward = PythonCall.pyconvert(Float64, py_reward)

        # display(info)
        # display(py_cheetah.unwrapped.data.qpos)
        # display(my_cheetah.data.qpos)

        Test.@testset "step" begin
            Test.@test reward ≈ py_reward atol=1e-5 rtol=1e-5
            Test.@test Environment.is_alive(my_cheetah) == !done
            Test.@test state ≈ py_state atol=1e-5 rtol=1e-5
        end
    end
end

# it should have space of 17 values
const HALF_CHEETAH_V5 = Dict{Symbol, Any}(
    :frame_skip => 5,
    :forward_reward_weight => 1.0,
    :ctrl_cost_weight => 0.1,
    :reset_noise_scale => 0.1,
    :exclude_current_positions_from_observation => true,
)

const HALF_CHEETAH_V1 = Dict{Symbol, Any}(
    :frame_skip => 5,
    :forward_reward_weight => 1.0,
    :ctrl_cost_weight => 0.1,
    :reset_noise_scale => 0.1,
    :exclude_current_positions_from_observation => true,
)

mutable struct HalfCheetah{R} <: AbstractEnvironment{NeuralNetwork.MatrixASSEQ} 
    # my params
    const max_steps::Int
    const seed::Int
    const reset_random_generator::Bool
    
    # general mutable params
    const model::MuJoCo.Model
    data_pool::MuJoCoDataPool
    data::Union{MuJoCo.Data, Nothing}
    random_generator::R
    current_step::Int

    # constants from gymnasium implementation
    const frame_skip::Int
    const forward_reward_weight::Float64
    const ctrl_cost_weight::Float64
    const reset_noise_scale::Float64
    const exclude_current_positions_from_observation::Bool
end

function get_environment(::Val{:HalfCheetah})::Type{HalfCheetah}
    return HalfCheetah
end


# xml file difference? look here: https://gymnasium.farama.org/environments/mujoco/humanoid/
function HalfCheetah(;
        base_version::Symbol=:half_cheetah_v5,
        xml_path=joinpath(DATA_DIR, "half_cheetah.xml"),
        seed::Int=rand(Int),
        reset_random_generator::Bool=true,
        max_steps::Int=1000,
        kwargs...
    )::HalfCheetah
    if base_version == :half_cheetah_v5
        base_dict = deepcopy(HALF_CHEETAH_V5)
    elseif base_version == :half_cheetah_v1
        base_dict = deepcopy(HALF_CHEETAH_V1)
    else
        throw("Unknown humanoid version, use :humanoid_v5 or :humanoid_v1, used $base_version")
    end

    for (key, value) in kwargs
        base_dict[key] = value
    end

    model, data_pool = get_model_data_pool(MUJOCO_MODEL_CACHE, xml_path)
    data = nothing
    current_steps = max_steps
    
    half_cheetah = HalfCheetah(
        max_steps,
        seed,
        reset_random_generator,

        model,
        data_pool,
        data,
        Random.Xoshiro(seed),
        current_steps,

        base_dict[:frame_skip],
        base_dict[:forward_reward_weight],
        base_dict[:ctrl_cost_weight],
        base_dict[:reset_noise_scale],
        base_dict[:exclude_current_positions_from_observation]
    )
    # I do not reset it for purpose - to reduce data footprint when I dont really use it!
    return half_cheetah
end

function visualize!(env::HalfCheetah, model::NeuralNetwork.AbstractAgentNeuralNetwork, parent_env=env, reset::Bool = true;)
    throw("unimplemented")
end

function get_action_size(env::HalfCheetah)::Int
    return 6
end

# function get_safe_data(env::HalfCheetah)::Dict{Symbol}
#     throw("unimplemented")
# end

# function load_safe_data!(env::HalfCheetah, data::Dict{Symbol}) 
#     throw("unimplemented")
# end

function reset!(env::HalfCheetah)
    if env.reset_random_generator
        env.random_generator = Random.Xoshiro(env.seed)
    end

    if isnothing(env.data)
        env.data = aquire!(env.data_pool)
    else
        MuJoCo.reset!(env.model, env.data)
    end
    _add_noise!(env)
    MuJoCo.forward!(env.model, env.data)
    env.current_step = 0   
end

function react!(env::HalfCheetah, actions::AbstractVector{Float32}) :: Float64
    for i in eachindex(actions)
        env.data.ctrl[i] = Float64(actions[i])
    end

    x_before = env.data.qpos[1]

    for _ in 1:env.frame_skip
        MuJoCo.step!(env.model, env.data)
    end
    MuJoCo.LibMuJoCo.mj_rnePostConstraint(env.model, env.data)

    env.current_step += 1
    x_after = env.data.qpos[1]
    dt = MuJoCo.timestep(env.model) * env.frame_skip
    x_vel = (x_after - x_before) / dt

    reward = _get_reward(env, x_vel)

    if !is_alive(env)
        release!(env.data_pool, env.data)
        env.data = nothing
    end
    return reward
end

function get_state(env::HalfCheetah) :: Vector{Float32}
    @assert !isnothing(env.data) "Data is not initialized, call reset!() first"

    observation_length = (
        ifelse(env.exclude_current_positions_from_observation, length(env.data.qpos) - 1, length(env.data.qpos)) +
        length(env.data.qvel)
    )

    observation = zeros(Float32, observation_length)
    index = 1
    for i in ifelse(env.exclude_current_positions_from_observation, 2, 1):length(env.data.qpos)
        observation[index] = Float32(env.data.qpos[i])
        index += 1
    end

    for i in 1:length(env.data.qvel)
        observation[index] = Float32(env.data.qvel[i])
        index += 1
    end

    return observation
end

function is_alive(env::HalfCheetah)::Bool
    return env.current_step < env.max_steps
end

function copy(env::HalfCheetah)
    if isnothing(env.data)
        new_data = nothing
    else
        new_data = aquire!(env.data_pool)
        MuJoCo.LibMuJoCo.mj_copyData(new_data, env.model, env.data)
    end
    return HalfCheetah(
        env.max_steps,
        env.seed,
        env.reset_random_generator,

        env.model,
        env.data_pool,
        new_data,
        Random.Xoshiro(env.seed),
        env.current_step,

        env.frame_skip,
        env.forward_reward_weight,
        env.ctrl_cost_weight,
        env.reset_noise_scale,
        env.exclude_current_positions_from_observation,
    )
end

# --------------------------------------------------------------------------------------------------------------------------------------
# INTERNAL

function _add_noise!(env::HalfCheetah)
    scale = env.reset_noise_scale
    rng = env.random_generator
    data = env.data
    if scale > 0.0
        for i in 1:length(data.qpos)
            data.qpos[i] += scale * 2 * (rand(rng, Float64) - 0.5)
        end

        for i in 1:length(data.qvel)
            data.qvel[i] += scale * randn(rng, Float64)
        end
    end
end

function _ctrl_cost(env::HalfCheetah)::Float64
    return env.ctrl_cost_weight * sum(abs2, env.data.ctrl)
end

function _get_reward(env::HalfCheetah, x_velocity::Float64)::Float64
    forward_reward = env.forward_reward_weight * x_velocity
    ctrl_cost = _ctrl_cost(env)

    # println("forward_reward: $forward_reward")
    # println("ctrl_cost: $ctrl_cost")

    reward = forward_reward - ctrl_cost
    return reward
end