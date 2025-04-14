export BasicCarEnvironment

# ------------------------------------------------------------------------------------------------

"Car dimmensions are half of the width and half of the height."
mutable struct BasicCarEnvironment{MB<:AbstractArray{Bool, 2}} <: AbstractEnvironment{NeuralNetwork.MatrixASSEQ}
    const map::MB
    const start_position::Tuple{Float64, Float64}
    const start_angle::Float64
    #const max_steps::Int
    max_steps::Int
    const angle_max_change::Float64
    const car_dimensions::Tuple{Float64, Float64}
    const initial_speed::Float64
    const min_speed::Float64
    const max_speed::Float64
    const speed_change::Float64
    const rays::Vector{Float64}
    const rays_distances_scale_factor::Float64
    const ray_input_clip::Float64
    const collision_reward::Float64

    # actual state:
    x::Float64
    y::Float64
    angle::Float64
    speed::Float64
    current_step::Int

    # tmp variables
    _is_alive::Bool
    const _corner_angle::Float64
    const _corner_distance::Float64
end


function get_environment(::Val{:BasicCarEnvironment})
    return BasicCarEnvironment
end


"""
Contructor for the BasicCarEnvironment struct.

All angles are in degrees, later they are converted to radians for calculations.
Car dimmensions should be full width and full height. They are converted to half width and half height in the constructor.
"""
function BasicCarEnvironment(;
    map::MB,
    start_position::Tuple{Float64, Float64},
    start_angle::Float64,
    max_steps::Int,
    angle_max_change::Float64,
    car_dimensions::Tuple{Float64, Float64},
    initial_speed::Float64,
    min_speed::Float64,
    max_speed::Float64,
    speed_change::Float64,
    rays::Vector{Float64},
    rays_distances_scale_factor::Float64,
    ray_input_clip::Float64,
    collision_reward::Float64
) :: BasicCarEnvironment where MB<:AbstractArray{Bool, 2}
    angle_max_change = deg2rad(angle_max_change)
    start_angle = deg2rad(start_angle)
    rays_rad = deg2rad.(rays)
    car_dimensions = (car_dimensions[1] / 2, car_dimensions[2] / 2)
    _corner_angle = atan(car_dimensions[1] / car_dimensions[2])
    _corner_distance = sqrt(car_dimensions[1]^2 + car_dimensions[2]^2)
    return BasicCarEnvironment(map, start_position, start_angle, max_steps, angle_max_change, car_dimensions, initial_speed, min_speed, max_speed, speed_change, rays_rad, rays_distances_scale_factor, ray_input_clip, collision_reward, start_position[1], start_position[2], start_angle, initial_speed, 0, true, _corner_angle, _corner_distance)
end

#--------------------------------------------------------------------------
# Interface functions

function copy(env::BasicCarEnvironment) :: BasicCarEnvironment
    # it should use named kwargs
    return BasicCarEnvironment(;
        map = env.map,
        start_position = env.start_position,
        start_angle = rad2deg(env.start_angle),
        max_steps = env.max_steps,
        angle_max_change = rad2deg(env.angle_max_change),
        car_dimensions = (env.car_dimensions[1] * 2, env.car_dimensions[2] * 2),
        initial_speed = env.initial_speed,
        min_speed = env.min_speed,
        max_speed = env.max_speed,
        speed_change = env.speed_change,
        rays = rad2deg.(env.rays),
        rays_distances_scale_factor = env.rays_distances_scale_factor,
        ray_input_clip = env.ray_input_clip,
        collision_reward = env.collision_reward
    )
end

function get_safe_data(env::BasicCarEnvironment)::Dict{Symbol}
    return Dict(
        :x => env.x,
        :y => env.y,
        :angle => env.angle,
        :speed => env.speed,
        :current_step => env.current_step
        :is_alive => env._is_alive
    )
end

function load_safe_data!(env::BasicCarEnvironment, data::Dict{Symbol})
    env.x = data[:x]
    env.y = data[:y]
    env.angle = data[:angle]
    env.speed = data[:speed]
    env.current_step = data[:current_step]
    env._is_alive = data[:is_alive]
end

function reset!(env::BasicCarEnvironment)
    env.x, env.y = env.start_position
    env.angle = env.start_angle
    env.current_step = 0
    env.speed = env.initial_speed
    env._is_alive = true
end

function react!(env::BasicCarEnvironment, action::AbstractVector{Float32}) :: Float64
    env.current_step += 1
    
    # for action space with 2 x 3 softmax outputs
    # steering_action = argmax(action[1:3])
    # speed_action = argmax(action[4:6])

    # for action space with 9 outputs, all possible combinations
    max_action = argmax(action)
    steering_action = (max_action - 1) % 3 + 1
    speed_action = (max_action - 1) ÷ 3 + 1  # 3 # (max_action - 1) ÷ 3 + 1

    if steering_action == 2
        env.angle += env.angle_max_change
    elseif steering_action == 3
        env.angle -= env.angle_max_change
    end

    if speed_action == 2
        env.speed += env.speed_change
    elseif speed_action == 3
        env.speed -= env.speed_change
    end

    env.speed = max(env.min_speed, min(env.max_speed, env.speed))
    env.angle = env.angle % (2 * pi)

    env.x = env.x + env.speed * cos(env.angle)
    env.y = env.y - env.speed * sin(env.angle)
    
    reward = (env.speed / env.max_speed) ^ 2
    if _does_collide(env, env.x, env.y)
        env._is_alive = false
        reward += env.collision_reward
    end

    return reward
end

function get_action_size(env::BasicCarEnvironment) :: Int
    return 9  # 9 # 3 # 6
end

# import Profile
# using PProf
# using BenchmarkTools
# function get_state(env::BasicCarEnvironment) :: Vector{Float32}
#     _get_state(env)

#     # b = BenchmarkTools.@benchmark _get_state($env)
#     # # b = BenchmarkTools.@benchmark _does_collide($env, $(env.x), $(env.y))
#     # display(b)
#     # throw("fvfddf")

#     Profile.clear()
#     Profile.@profile for i in 1:1000000
#         _get_state(env)
#     end
#     pprof(;webport=2137)
#     sleep(100)
# end

function get_state(env::BasicCarEnvironment) :: Vector{Float32}
    inputs = Vector{Float32}(undef, length(env.rays) + 1)
    for i in eachindex(env.rays)
        ray_distance = _get_ray_distance(env, env.angle + env.rays[i]) / env.rays_distances_scale_factor
        inputs[i] = ray_distance > env.ray_input_clip ? env.ray_input_clip : ray_distance
    end
    speed_input = env.speed / env.max_speed
    inputs[end] = speed_input
    return inputs
end

function is_alive(env::BasicCarEnvironment)
    return env.current_step < env.max_steps && env._is_alive
end


#--------------------------------------------------------------------------
# Protected functions

"Return True if collide with the wall, False otherwise."
function _does_collide(env::BasicCarEnvironment, x, y) :: Bool
    return (
        _does_collide_at_position(env, x, y) ||
        _does_collide_at_angle(env, env._corner_distance, env.angle + env._corner_angle) ||
        _does_collide_at_angle(env, env._corner_distance, env.angle - env._corner_angle) ||
        _does_collide_at_angle(env, env._corner_distance, env.angle + env._corner_angle + pi) ||
        _does_collide_at_angle(env, env._corner_distance, env.angle - env._corner_angle + pi) ||
        _does_collide_at_angle(env, env.car_dimensions[1], env.angle + pi/2) ||
        _does_collide_at_angle(env, env.car_dimensions[1], env.angle - pi/2) ||
        _does_collide_at_angle(env, env.car_dimensions[2], env.angle) ||
        _does_collide_at_angle(env, env.car_dimensions[2], env.angle + pi)
    )
end

function _does_collide_at_angle(env::BasicCarEnvironment, distance::Float64, angle::Float64) :: Bool
    x_check = unsafe_trunc(Int, env.x + distance * cos(angle) + 0.5)
    y_check = unsafe_trunc(Int, env.y - distance * sin(angle) + 0.5)

    if x_check < 1 || x_check > size(env.map, 2) || y_check < 1 || y_check > size(env.map, 1)
        return true
    end
    return @inbounds env.map[y_check, x_check]
end

function _does_collide_at_position(env::BasicCarEnvironment, x::Float64, y::Float64) :: Bool
    return _does_collide_at_position_faster(env, x + 0.5, y + 0.5)
end

@inline function _does_collide_at_position_faster(env::BasicCarEnvironment, x::Float64, y::Float64) :: Bool
    x_check = unsafe_trunc(Int, x)  # we assume that x and y are not inf or nan
    y_check = unsafe_trunc(Int, y)

    if x_check < 1 || x_check > size(env.map, 2) || y_check < 1 || y_check > size(env.map, 1)
        return true
    end

    return @inbounds env.map[y_check, x_check]
end

const INITIAL_RAY_STEP = 25.0
const STEP_MULTIPLIER = 0.2  # should be below 1.0

function _get_ray_distance(env::BasicCarEnvironment, angle::Float64) :: Float64
    x = env.x + 0.5
    y = env.y + 0.5
    distance = 0.0
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    max_distance = env.ray_input_clip * env.rays_distances_scale_factor

    x_prev = x
    y_prev = y
    step_course = INITIAL_RAY_STEP

    while true
        @fastmath x += cos_angle * step_course
        @fastmath y -= sin_angle * step_course
        @fastmath distance += step_course
        if _does_collide_at_position_faster(env, x, y)
            if step_course == 1.0
                return distance
            end

            distance -= step_course
            x = x_prev
            y = y_prev
            
            @fastmath step_course = max(round(step_course * STEP_MULTIPLIER), 1.0)
        else
            x_prev = x
            y_prev = y
        end

        if distance > max_distance
            return max_distance
        end
    end
end

# include visualization function
include("_CarEnvironment_visualize.jl")
