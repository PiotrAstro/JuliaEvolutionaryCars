# this is a translation of Farama-Foundation Gymnasium Humanoid implementation to pure Julia environment
# their implementation: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/humanoid_v5.py


# ---------------------------------------------------------------------------------------------
# MuJoCo data pools - it reduces memory problems
# it is universal for all mujoco models

# should set it correctly, it is just a guess, for very powerfull machine I should set it higher
const MAX_DATA_POOL_SIZE = min(1000, Threads.nthreads() * 10)

struct MuJoCoDataPool
    locker::ReentrantLock
    model::MuJoCo.Model
    data::Vector{Union{MuJoCo.Data, Nothing}}
    free::BitVector
end

function MuJoCoDataPool(model::MuJoCo.Model, max_size::Int=MAX_DATA_POOL_SIZE)::MuJoCoDataPool
    return MuJoCoDataPool(
        ReentrantLock(),
        model,
        Vector{Union{MuJoCo.Data, Nothing}}(nothing, max_size),
        trues(max_size)
    )
end

"""
No matter the cache size, it will return already reset data object.
It is thread safe.
Returned objects are reseted.
"""
function aquire!(data_pool::MuJoCoDataPool)::MuJoCo.Data
    data_object::Union{MuJoCo.Data, Nothing} = lock(data_pool.locker) do
        index = findfirst(data_pool.free)
        if isnothing(index)
            return nothing
        else
            data_pool.free[index] = false
            return data_pool.data[index]
        end
    end
    if isnothing(data_object)
        data_object = MuJoCo.init_data(data_pool.model)
    else
        MuJoCo.reset!(data_pool.model, data_object)
    end
    return data_object
end

"""
It will put data object back to the pool.
It is thread safe.
"""
function release!(data_pool::MuJoCoDataPool, data_object::MuJoCo.Data)
    lock(data_pool.locker) do
        index = findfirst(!, data_pool.free)
        if !isnothing(index)
            data_pool.data[index] = data_object
            data_pool.free[index] = true
        end
    end
end

struct MuJoCoModelCache
    locker::ReentrantLock
    cache::Dict{String, Tuple{MuJoCo.Model, MuJoCoDataPool}}
end

"""
It is thread safe.
"""
function get_model_data_pool(model_cache::MuJoCoModelCache, xml_path::String)::Tuple{MuJoCo.Model, MuJoCoDataPool}
    return lock(model_cache.locker) do
        if haskey(model_cache.cache, xml_path)
            return model_cache.cache[xml_path]
        else
            model = MuJoCo.load_model(xml_path)
            data_pool = MuJoCoDataPool(model)
            model_cache.cache[xml_path] = (model, data_pool)
            return (model, data_pool)
        end
    end
end

const MUJOCO_MODEL_CACHE = MuJoCoModelCache(ReentrantLock(), Dict{String, Tuple{MuJoCo.Model, MuJoCoDataPool}}())







# ---------------------------------------------------------------------------------------------------------------------------------------
# Humanoid environment




@testitem "Humanoid-v5" begin
    import PythonCall
    import JuliaEvolutionaryCars.Environment as Environment
    import Test

    gym = PythonCall.pyimport("gymnasium")
    py_humanoid = gym.make("Humanoid-v5", reset_noise_scale=0.0, terminate_when_unhealthy=false)
    py_humanoid.reset()
    my_humanoid = Environment.Humanoid(
        base_version=:humanoid_v5,
        reset_noise_scale=0.0,
        terminate_when_unhealthy=false,
        max_steps=400
    )
    my_humanoid.reset()

    random_nn = Environment.NeuralNetwork.Random_NN(Environment.get_action_size(my_humanoid))
    while Environment.is_alive(my_humanoid) && my_humanoid.current_step < my_humanoid.max_steps -1
        action = Environment.NeuralNetwork.predict(random_nn, Environment.NeuralNetwork.MatrixASSEQ([ones(Float32, 1)]))[:, 1]
        # action .= 0.0f0
        action = clamp.(action, -0.4f0, 0.4f0)
        reward = Environment.react!(my_humanoid, action)
        state = Environment.get_state(my_humanoid)
        py_state, py_reward, done, _, info = py_humanoid.step(action)
        done = PythonCall.pyconvert(Bool, done)
        py_state = PythonCall.pyconvert(Vector{Float32}, py_state)
        py_reward = PythonCall.pyconvert(Float64, py_reward)

        Test.@testset "step" begin
            Test.@test reward ≈ py_reward atol=1e-5 rtol=1e-5
            Test.@test Environment.is_alive(my_humanoid) == !done
            Test.@test state ≈ py_state atol=1e-5 rtol=1e-5
        end
    end
end

# this one should have space of 348 values
const HUMANOID_V5 = Dict{Symbol, Any}(
    :frame_skip => 5,
    :forward_reward_weight => 1.25,
    :ctrl_cost_weight => 0.1,
    :contact_cost_weight => 5e-7,
    :contact_cost_range => (nextfloat(typemin(Float64)), 10.0),
    :healthy_reward => 5.0,
    :terminate_when_unhealthy => true,
    :healthy_z_range => (1.0, 2.0),
    :reset_noise_scale => 1e-2,
    :exclude_current_positions_from_observation => true,
    :exclude_first_cinert_in_observation => 1,  # -1 means it will be ommited althoughether
    :exclude_first_cvel_in_observation => 1,
    :exclude_first_qfrc_in_observation => 6,
    :exclude_first_cfrc_in_observation => 1
)

# I guess 376 values
const HUMANOID_V4 = Dict{Symbol, Any}(
    :frame_skip => 5,
    :forward_reward_weight => 1.25,
    :ctrl_cost_weight => 0.1,
    :contact_cost_weight => 0.0,
    :contact_cost_range => (nextfloat(typemin(Float64)), 10.0),
    :healthy_reward => 5.0,
    :terminate_when_unhealthy => true,
    :healthy_z_range => (1.0, 2.0),
    :reset_noise_scale => 1e-2,
    :exclude_current_positions_from_observation => true,
    :exclude_first_cinert_in_observation => 0,  # -1 means it will be ommited althoughether
    :exclude_first_cvel_in_observation => 0,
    :exclude_first_qfrc_in_observation => 0,
    :exclude_first_cfrc_in_observation => 0
)

# I guess 376 values
const HUMANOID_V1 = Dict{Symbol, Any}(
    :frame_skip => 5,
    :forward_reward_weight => 1.25,
    :ctrl_cost_weight => 0.1,
    :contact_cost_weight => 5e-7,
    :contact_cost_range => (nextfloat(typemin(Float64)), 10.0),
    :healthy_reward => 5.0,
    :terminate_when_unhealthy => true,
    :healthy_z_range => (1.0, 2.0),
    :reset_noise_scale => 1e-2,
    :exclude_current_positions_from_observation => true,
    :exclude_first_cinert_in_observation => 0,  # -1 means it will be ommited althoughether
    :exclude_first_cvel_in_observation => 0,
    :exclude_first_qfrc_in_observation => 0,
    :exclude_first_cfrc_in_observation => 0
)

mutable struct Humanoid{R} <: AbstractEnvironment{NeuralNetwork.MatrixASSEQ} 
    # my params
    const max_steps::Int
    const seed::Int
    const reset_random_generator::Bool
    const additional_normalization::Bool
    
    # general mutable params
    const model::MuJoCo.Model
    const data_pool::MuJoCoDataPool
    data::Union{MuJoCo.Data, Nothing}
    random_generator::R
    is_healthy::Bool
    current_step::Int

    # constants from gymnasium implementation
    const frame_skip::Int
    const forward_reward_weight::Float64
    const ctrl_cost_weight::Float64
    const contact_cost_weight::Float64
    const contact_cost_range::Tuple{Float64, Float64}
    const healthy_reward::Float64
    const terminate_when_unhealthy::Bool
    const healthy_z_range::Tuple{Float64, Float64}
    const reset_noise_scale::Float64
    const exclude_current_positions_from_observation::Bool
    const exclude_first_cinert_in_observation::Int
    const exclude_first_cvel_in_observation::Int
    const exclude_first_qfrc_in_observation::Int
    const exclude_first_cfrc_in_observation::Int
end

function get_environment(::Val{:Humanoid})::Type{Humanoid}
    return Humanoid
end

# xml file difference? look here: https://gymnasium.farama.org/environments/mujoco/humanoid/
function Humanoid(;
        base_version::Symbol=:humanoid_v5,
        additional_normalization::Bool=false,  # if false, then ser should provide actions in range (-0.4, 0.4), if true then (-1, 1) and we will normalize it internally
        xml_path=joinpath(DATA_DIR, "humanoid-post0.21.xml"),  # dont like it, maybe my main module should export data path?
        kwargs...
    )::Humanoid
    if base_version == :humanoid_v5
        base_dict = deepcopy(HUMANOID_V5)
    elseif base_version == :humanoid_v4
        base_dict = deepcopy(HUMANOID_V4)
    elseif base_version == :humanoid_v1
        base_dict = deepcopy(HUMANOID_V1)
    else
        throw("Unknown humanoid version, use :humanoid_v5 or :humanoid_v1, used $base_version")
    end

    for (key, value) in kwargs
        base_dict[key] = value
    end

    model, data_pool = get_model_data_pool(MUJOCO_MODEL_CACHE, xml_path)
    data = nothing
    seed = get(base_dict, :seed, rand(Int))
    reset_random_generator = get(base_dict, :reset_random_generator, true)
    max_steps = get(base_dict, :max_steps, 1000)
    current_step = max_steps
    
    humanoid = Humanoid(
        max_steps,
        seed,
        reset_random_generator,
        additional_normalization,

        model,
        data_pool,
        data,
        Random.Xoshiro(seed),
        true,
        current_step,

        base_dict[:frame_skip],
        base_dict[:forward_reward_weight],
        base_dict[:ctrl_cost_weight],
        base_dict[:contact_cost_weight],
        base_dict[:contact_cost_range],
        base_dict[:healthy_reward],
        base_dict[:terminate_when_unhealthy],
        base_dict[:healthy_z_range],
        base_dict[:reset_noise_scale],
        base_dict[:exclude_current_positions_from_observation],
        base_dict[:exclude_first_cinert_in_observation],
        base_dict[:exclude_first_cvel_in_observation],
        base_dict[:exclude_first_qfrc_in_observation],
        base_dict[:exclude_first_cfrc_in_observation]
    )
    # I do not reset it on purpose - to not allocate data object!
    return humanoid
end

function visualize!(env::Humanoid, model::NeuralNetwork.AbstractAgentNeuralNetwork, parent_env=env, reset::Bool = true;)
    throw("unimplemented")
end

function get_action_size(env::Humanoid)::Int
    return 17
end

# function get_safe_data(env::Humanoid)::Dict{Symbol}
#     throw("unimplemented")
# end

# function load_safe_data!(env::Humanoid, data::Dict{Symbol}) 
#     throw("unimplemented")
# end

function reset!(env::Humanoid)
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
    env.is_healthy = true
    env.current_step = 0
end

function react!(env::Humanoid, actions::AbstractVector{Float32}) :: Float64
    @assert length(actions) == length(env.data.ctrl)
    @assert is_alive(env) "You have to reset the environment first!"

    for i in eachindex(actions)
        env.data.ctrl[i] = Float64(ifelse(env.additional_normalization, actions[i] * 0.4f0, actions[i]))
    end

    xy_before = _mass_center(env)

    for _ in 1:env.frame_skip
        MuJoCo.step!(env.model, env.data)
    end
    MuJoCo.LibMuJoCo.mj_rnePostConstraint(env.model, env.data)

    env.current_step += 1
    xy_after = _mass_center(env)
    dt = MuJoCo.timestep(env.model) * env.frame_skip
    x_vel = (xy_after[1] - xy_before[1]) / dt
    y_vel = (xy_after[2] - xy_before[2]) / dt

    env.is_healthy = _is_healthy(env)
    reward = _get_reward(env, x_vel)

    if !is_alive(env)
        release!(env.data_pool, env.data)
        env.data = nothing
    end
    
    return reward
end

function get_state(env::Humanoid) :: Vector{Float32}
    @assert !isnothing(env.data) "You have to reset the environment first!"
    observation_length = (
        ifelse(env.exclude_current_positions_from_observation, length(env.data.qpos) - 2, length(env.data.qpos)) +
        length(env.data.qvel) +
        ifelse(env.exclude_first_cinert_in_observation == -1, 0, length(env.data.cinert) - env.exclude_first_cinert_in_observation * size(env.data.cinert, 2)) +
        ifelse(env.exclude_first_cvel_in_observation == -1, 0, length(env.data.cvel) - env.exclude_first_cvel_in_observation * size(env.data.cvel, 2)) +
        ifelse(env.exclude_first_qfrc_in_observation == -1, 0, length(env.data.qfrc_actuator) - env.exclude_first_qfrc_in_observation * size(env.data.qfrc_actuator, 2)) +
        ifelse(env.exclude_first_cfrc_in_observation == -1, 0, length(env.data.cfrc_ext) - env.exclude_first_cfrc_in_observation * size(env.data.cfrc_ext, 2))
    )

    observation = zeros(Float32, observation_length)
    index = 1
    for i in ifelse(env.exclude_current_positions_from_observation, 3, 1):length(env.data.qpos)
        observation[index] = Float32(env.data.qpos[i])
        index += 1
    end

    for i in 1:length(env.data.qvel)
        observation[index] = Float32(env.data.qvel[i])
        index += 1
    end

    if env.exclude_first_cinert_in_observation != -1
        for row in (1+env.exclude_first_cinert_in_observation):size(env.data.cinert, 1)
            for col in 1:size(env.data.cinert, 2)
                observation[index] = Float32(env.data.cinert[row, col])
                index += 1
            end
        end
    end

    if env.exclude_first_cvel_in_observation != -1
        for row in (1+env.exclude_first_cvel_in_observation):size(env.data.cvel, 1)
            for col in 1:size(env.data.cvel, 2)
                observation[index] = Float32(env.data.cvel[row, col])
                index += 1
            end
        end
    end

    if env.exclude_first_qfrc_in_observation != -1
        for row in (1+env.exclude_first_qfrc_in_observation):size(env.data.qfrc_actuator, 1)
            for col in 1:size(env.data.qfrc_actuator, 2)
                observation[index] = Float32(env.data.qfrc_actuator[row, col])
                index += 1
            end
        end
    end

    if env.exclude_first_cfrc_in_observation != -1
        for row in (1+env.exclude_first_cfrc_in_observation):size(env.data.cfrc_ext, 1)
            for col in 1:size(env.data.cfrc_ext, 2)
                observation[index] = Float32(env.data.cfrc_ext[row, col])
                index += 1
            end
        end
    end

    return observation
end

function is_alive(env::Humanoid)::Bool
    return (env.is_healthy || !env.terminate_when_unhealthy) && env.current_step < env.max_steps
end

function copy(env::Humanoid)
    if isnothing(env.data)
        new_data = nothing
    else
        new_data = aquire!(env.data_pool)
        MuJoCo.LibMuJoCo.mj_copyData(new_data, env.model, env.data)
    end
    return Humanoid(
        env.max_steps,
        env.seed,
        env.reset_random_generator,
        env.additional_normalization,

        env.model,
        env.data_pool,
        new_data,
        Random.Xoshiro(env.seed),
        env.is_healthy,
        env.current_step,

        env.frame_skip,
        env.forward_reward_weight,
        env.ctrl_cost_weight,
        env.contact_cost_weight,
        env.contact_cost_range,
        env.healthy_reward,
        env.terminate_when_unhealthy,
        env.healthy_z_range,
        env.reset_noise_scale,
        env.exclude_current_positions_from_observation,
        env.exclude_first_cinert_in_observation,
        env.exclude_first_cvel_in_observation,
        env.exclude_first_qfrc_in_observation,
        env.exclude_first_cfrc_in_observation
    )
end

# --------------------------------------------------------------------------------------------------------------------------------------
# INTERNAL

function _mass_center(env::Humanoid)::Tuple{Float64, Float64}
    mass_sum = sum(env.model.body_mass)
    first_axis = LinearAlgebra.dot(view(env.model.body_mass, :, 1), view(env.data.xipos, :, 1)) / mass_sum
    second_axis = LinearAlgebra.dot(view(env.model.body_mass, :, 1), view(env.data.xipos, :, 2)) / mass_sum
    return first_axis, second_axis
end

function _add_noise!(env::Humanoid)
    scale = env.reset_noise_scale
    rng = env.random_generator
    data = env.data
    if scale > 0.0
        for i in eachindex(data.qpos)
            data.qpos[i] += scale * 2 * (rand(rng, Float64) - 0.5)
        end

        for i in eachindex(data.qvel)
            data.qvel[i] += scale * 2 * (randn(rng, Float64) - 0.5)
        end
    end
end

function _healthy_reward(env::Humanoid)::Float64
    return env.healthy_reward * env.is_healthy
end

function _ctrl_cost(env::Humanoid)::Float64
    return env.ctrl_cost_weight * sum(abs2, env.data.ctrl)
end

function _contact_cost(env::Humanoid)::Float64
    contact_cost = env.contact_cost_weight * sum(abs2, env.data.cfrc_ext)
    return clamp(contact_cost, env.contact_cost_range[1], env.contact_cost_range[2])
end

function _is_healthy(env::Humanoid)::Bool
    z = env.data.qpos[3]
    return z > env.healthy_z_range[1] && z < env.healthy_z_range[2]
end

function _get_reward(env::Humanoid, x_velocity::Float64)::Float64
    forward_reward = env.forward_reward_weight * x_velocity
    healthy_reward = _healthy_reward(env)

    ctrl_cost = _ctrl_cost(env)
    contact_cost = _contact_cost(env)

    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    return reward
end

