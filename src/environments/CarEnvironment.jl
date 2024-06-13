module CarEnvironment
    include("Environment.jl")
    include("../neural_network/NeuralNetwork.jl")
    using .NeuralNetwork
    using .Environment
    using Gtk
    export BasicCarEnvironment, get_safe_data, load_safe_data!, reset!, react!, get_state, is_alive

    "Car dimmensions are half of the width and half of the height."
    mutable struct BasicCarEnvironment <: AbstractEnvironment
        const map_view::Matrix{Bool}
        const start_position::Tuple{Float64, Float64}
        const start_angle::Float64
        const max_steps::Int
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
        const _2_corner_angles::Tuple{Float64, Float64}
        const _corner_distance::Float64
    end


    """
    Contructor for the BasicCarEnvironment struct.

    All angles are in degrees, later they are converted to radians for calculations.
    Car dimmensions should be full width and full height. They are converted to half width and half height in the constructor.
    """
    function BasicCarEnvironment(;
                                map_view::Matrix{Bool},
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
                                ) :: BasicCarEnvironment
        angle_max_change = deg2rad(angle_max_change)
        start_angle = deg2rad(start_angle)
        rays = deg2rad.(rays)
        car_dimensions = (car_dimensions[1] / 2, car_dimensions[2] / 2)
        _2_corner_angles = (atan(car_dimensions[2] / car_dimensions[1]), atan(car_dimensions[1] / car_dimensions[2]))
        _corner_distance = sqrt(car_dimensions[1]^2 + car_dimensions[2]^2)
        return BasicCarEnvironment(map_view, start_position, start_angle, max_steps, angle_max_change, car_dimensions, initial_speed, min_speed, max_speed, speed_change, rays, rays_distances_scale_factor, ray_input_clip, collision_reward, start_position[1], start_position[2], start_angle, initial_speed, 0, true, _2_corner_angles, _corner_distance)
    end

    #--------------------------------------------------------------------------
    # Interface functions

    "resets the environment before, doesnt reset after, concrete implementation will have some kwargs"
    function visulize!(env::AbstractEnvironment, model::NeuralNetwork.AbstractNeuralNetwork; car_path::String, map_path::String)
        reset!(env)

        win = GtkWindow("Car Simulation", size(env.map, 1), size(env.map, 2))
        canvas = @GtkCanvas()
        push!(win, canvas)

        car_image = Gtk.GdkPixbuf.Pixbuf(car_path)
        map_image = Gtk.GdkPixbuf.Pixbuf(map_path)

        function draw_scene()
            # Draw the map first
            ctx = getgc(canvas)
            Gtk.set_source_pixbuf(canvas, map_image, 0, 0)
            paint(canvas)
        
            # Draw the car on top of the map
            save(canvas) do
                translate(cr, car.position[1], car.position[2])
                rotate(cr, env.angle)
                translate(cr, -car.dimensions[1] / 2, -car.dimensions[2] / 2)
                Gtk.set_source_pixbuf(cr, car.image, 0, 0)
                paint(cr)
            end
        end

        function on_draw(widget, cr)
            set_source_rgb(cr, 0, 0, 0)
            paint(cr)
            draw_scene(cr, car, map_image)
            return true
        end



        Gtk.cairo_set_source_pixbuf(canvas, map_image, 0, 0)
        paint(cr)
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

    function react!(env::BasicCarEnvironment, action) :: Float64
        env.current_step += 1
        
        steering_action = max(action[1:3])
        speed_action = max(action[4:5])

        if steering_action == 1
            env.angle += env.angle_max_change
        elseif steering_action == 2
            env.angle -= env.angle_max_change
        end

        if speed_action == 1
            env.speed += env.speed_change
        elseif speed_action == 2
            env.speed -= env.speed_change
        end

        env.speed = max(env.min_speed, min(env.max_speed, env.speed))
        env.angle = env.angle % (2 * pi)

        env.x = env.x + env.speed * cos(env.angle)
        env.y = env.y + env.speed * sin(env.angle)
        
        reward = (env.speed / env.max_speed) ^ 2
        if _does_collide(env, env.x, env.y)
            env._is_alive = false
            reward += env.collision_reward
        end

        return reward
    end

    function get_state(env::BasicCarEnvironment) :: Array{Float32}
        inputs = Vector{Float64}([_get_ray_distance(env, env.angle + ray) for ray in env.rays])
        inputs = inputs ./ env.rays_distances_scale_factor
        inputs = inputs .> env.ray_input_clip ? env.ray_input_clip : inputs
        speed_input = env.speed / env.max_speed
        push!(inputs, speed_input)
        return Array{Float32}(inputs)
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
            _does_collide_at_angle(env, env._corner_distance, env.angle + env._2_corner_angles[1]) ||
            _does_collide_at_angle(env, env._corner_distance, env.angle - env._2_corner_angles[1]) ||
            _does_collide_at_angle(env, env._corner_distance, env.angle + env._2_corner_angles[2] + pi) ||
            _does_collide_at_angle(env, env._corner_distance, env.angle - env._2_corner_angles[2] + pi) ||
            _does_collide_at_angle(env, env.car_dimensions[1], env.angle + pi/2) ||
            _does_collide_at_angle(env, env.car_dimensions[1], env.angle - pi/2) ||
            _does_collide_at_angle(env, env.car_dimensions[2], env.angle) ||
            _does_collide_at_angle(env, env.car_dimensions[2], env.angle + pi)
        )
    end

    function _does_collide_at_angle(env::BasicCarEnvironment, distance::Float64, angle::Float64) :: Bool
        x_check = Int(round(env.x + distance * cos(angle)))
        y_check = Int(round(env.y + distance * sin(angle)))

        if x_check < 0 || x_check >= size(env.map_view, 1) || y_check < 0 || y_check >= size(env.map_view, 2)
            return true
        end
        return env.map_view[x_check, y_check]
    end

    function _does_collide_at_position(env::BasicCarEnvironment, x::Float64, y::Float64) :: Bool
        x_check = Int(round(x))
        y_check = Int(round(y))

        if x_check < 0 || x_check >= size(env.map_view, 1) || y_check < 0 || y_check >= size(env.map_view, 2)
            return true
        end
        return env.map_view[x_check, y_check]
    end

    function _get_ray_distance(env::BasicCarEnvironment, angle::Float64) :: Float64
        x = env.x
        y = env.y
        distance = 0.0
        max_distance = env.ray_input_clip * env.rays_distances_scale_factor
        while distance < max_distance
            x += cos(angle)
            y += sin(angle)
            distance += 1.0
            if _does_collide_at_position(env, x, y)
                return distance
            end
        end
        return distance
    end
end
