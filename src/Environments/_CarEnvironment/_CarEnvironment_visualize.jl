

function visualize!(env::BasicCarEnvironment, model::NeuralNetwork.AbstractAgentNeuralNetwork, parent_env=env, reset::Bool = true; car_image_path::String, map_image_path::String, fps::Int = 60, log::Bool = false)
    if reset
        reset!(parent_env)
    end

    SDL.SDL_GL_SetAttribute(SDL.SDL_GL_MULTISAMPLEBUFFERS, 1)
    SDL.SDL_GL_SetAttribute(SDL.SDL_GL_MULTISAMPLESAMPLES, 4)

    @assert SDL.SDL_Init(SDL.SDL_INIT_EVERYTHING) == 0 "error initializing SDL: $(unsafe_string(SDL.SDL_GetError()))"

    win = SDL.SDL_CreateWindow("Car Simulation", SDL.SDL_WINDOWPOS_CENTERED, SDL.SDL_WINDOWPOS_CENTERED, size(env.map, 2), size(parent_env.map, 1), SDL.SDL_WINDOW_OPENGL)
    SDL.SDL_SetWindowResizable(win, SDL.SDL_TRUE)

    renderer = SDL.SDL_CreateRenderer(win, -1, SDL.SDL_RENDERER_ACCELERATED | SDL.SDL_RENDERER_PRESENTVSYNC)

    car_surface = LSDL2.IMG_Load(car_image_path)
    car_tex = SDL.SDL_CreateTextureFromSurface(renderer, car_surface)
    SDL.SDL_FreeSurface(car_surface)

    map_surface = LSDL2.IMG_Load(map_image_path)
    map_tex = SDL.SDL_CreateTextureFromSurface(renderer, map_surface)
    SDL.SDL_FreeSurface(map_surface)

    car_w_ref, car_h_ref = Ref{Cint}(0), Ref{Cint}(0)
    SDL.SDL_QueryTexture(car_tex, C_NULL, C_NULL, car_w_ref, car_h_ref)
    car_w, car_h = car_w_ref[], car_h_ref[]

    map_w_ref, map_h_ref = Ref{Cint}(0), Ref{Cint}(0)
    SDL.SDL_QueryTexture(map_tex, C_NULL, C_NULL, map_w_ref, map_h_ref)
    map_w, map_h = map_w_ref[], map_h_ref[]

    try
        close = false
        counter = 0
        time_start = time()
        action = zeros(Float32, get_action_size(parent_env))
        while !close
            event_ref = Ref{SDL.SDL_Event}()
            while Bool(SDL.SDL_PollEvent(event_ref))
                evt = event_ref[]
                evt_ty = evt.type
                if evt_ty == SDL.SDL_QUIT
                    close = true
                    break
                end
                
                if isa(model, NeuralNetwork.DummyNN)
                    if evt_ty == SDL.SDL_KEYDOWN
                        key = evt.key.keysym.sym
                        if key == LSDL2.SDLK_w || key == LSDL2.SDLK_UP
                            action[4:6] .= 0.0
                            action[5] = 1.0
                        elseif key == LSDL2.SDLK_s || key == LSDL2.SDLK_DOWN
                            action[4:6] .= 0.0
                            action[6] = 1.0
                        end

                        if key == LSDL2.SDLK_a || key == LSDL2.SDLK_LEFT
                            action[1:3] .= 0.0
                            action[2] = 1.0
                        elseif key == LSDL2.SDLK_d || key == LSDL2.SDLK_RIGHT
                            action[1:3] .= 0.0
                            action[3] = 1.0
                        end
                    elseif evt_ty == SDL.SDL_KEYUP
                        key = evt.key.keysym.sym
                        if key == LSDL2.SDLK_w || key == LSDL2.SDLK_UP
                            action[5] = 0.0
                        end
                        if key == LSDL2.SDLK_s || key == LSDL2.SDLK_DOWN
                            action[6] = 0.0
                        end

                        if key == LSDL2.SDLK_a || key == LSDL2.SDLK_LEFT
                            action[2] = 0.0
                        end
                        if key == LSDL2.SDLK_d || key == LSDL2.SDLK_RIGHT
                            action[3] = 0.0
                        end
                    end
                end
            end

            if is_alive(parent_env)
                state = reshape(get_state(parent_env), :, 1)
                action = isa(model, NeuralNetwork.DummyNN) ? action : NeuralNetwork.predict(model, state)[:, 1]
                react!(parent_env, action)
            else
                close = true
            end

            car_pos = (round(Int, env.x), round(Int, env.y))
            car_angle = env.angle  # Assuming you have this function to get the car's angle
            car_x, car_y = car_pos[1], car_pos[2]
            car_dim = round.(Int, env.car_dimensions .* 2)  # Assuming env.dimensions returns a tuple (width, height)
            car_w, car_h = car_dim[1], car_dim[2]

            # Destination rectangle for the car texture (positioned around the center)
            car_dest_ref = Ref(LSDL2.SDL_Rect(car_x - car_h ÷ 2, car_y - car_w ÷ 2, car_h, car_w))
            # car_dest_ref = Ref(LSDL2.SDL_Rect(car_x, car_y, car_h, car_w))

            # Destination rectangle for the map texture
            map_dest_ref = Ref(LSDL2.SDL_Rect(0, 0, map_w, map_h))

            SDL.SDL_RenderClear(renderer)
            
            # Render the map texture to cover the entire window
            SDL.SDL_RenderCopy(renderer, map_tex, C_NULL, map_dest_ref)
            
            # Render the car texture at the car's position with rotation, rotating around its center
            center = Ref(LSDL2.SDL_Point(car_h ÷ 2, car_w ÷ 2))
            SDL.SDL_RenderCopyEx(renderer, car_tex, C_NULL, car_dest_ref, -rad2deg(car_angle), center, SDL.SDL_FLIP_NONE)
            
            SDL.SDL_RenderPresent(renderer)

            SDL.SDL_Delay(1000 ÷ fps)

            if log
                println("actions: ", action)
                counter += 1
                if time() - time_start >= 1.0
                    time_end = time()
                    println("FPS: ", counter / (time_end - time_start))
                    counter = 0
                    time_start = time()
                end
            end
        end
    catch e
        println("Error: ", e)
    finally
        SDL.SDL_DestroyTexture(car_tex)
        SDL.SDL_DestroyTexture(map_tex)
        SDL.SDL_DestroyRenderer(renderer)
        SDL.SDL_DestroyWindow(win)
        SDL.SDL_Quit()
    end
end