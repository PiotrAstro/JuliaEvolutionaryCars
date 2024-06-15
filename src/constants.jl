using Images
using FileIO

map_image_path = raw"data\map2.png"
car_image_path = raw"data\car.png"
map_image_path_2 = raw"data\map.png"

img = Gray.(load(map_image_path))
map = Matrix{Bool}(Array(img) .> 0.5)

img2 = Gray.(load(map_image_path_2))
map_2 = Matrix{Bool}(Array(img2) .> 0.5)

MAX_STEPS = 2000

CONSTANTS_DICT = Dict(
    :run_config => Dict(
        :max_generations => 1000,
        :max_evaluations => 100000,
        :log => true,
        :visualize_each_n_epochs => 50,
    ),
    :environment => Dict(
        :name => :BasicCarEnvironment,
        :visualization => Dict(
            :car_image_path => car_image_path,
            :map_image_path => map_image_path,
            :fps => 60,
        ),
        :universal_kwargs => Dict(
            :angle_max_change => 1.15,  # 1.15
            :car_dimensions => (30.0, 45.0),  # width, height
            :initial_speed => 1.2,  # 1.2
            :min_speed => 1.2,  # 1.2
            :max_speed => 6.0,
            :speed_change => 0.04,
            :rays => [-90.0, -67.5, -45.0, -22.5, 0.0, 22.5, 45.0, 67.5, 90.0],  # (-90, -45, 0, 45, 90),  # (-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90)
            :rays_distances_scale_factor => 100.0,
            :ray_input_clip => 5.0,
            :collision_reward => -100.0
        ),
        :changeable_training_kwargs_list => [
            Dict(
                :map => map,
                :start_position => (504.0, 744.0),
                :start_angle => 0.0,
                :max_steps => MAX_STEPS
            ),
            Dict(
                :map => map,
                :start_position => (504.0, 744.0),
                :start_angle => 180.0,
                :max_steps => MAX_STEPS
            ),
            Dict(
                :map => map,
                :start_position => (425.0, 337.0),
                :start_angle => 170.0,
                :max_steps => MAX_STEPS
            ),
            Dict(
                :map => map,
                :start_position => (283.0, 536.0),
                :start_angle => 200.0,
                :max_steps => MAX_STEPS
            ),
            Dict(
                :map => map,
                :start_position => (665.0, 400.0),
                :start_angle => 270.0,
                :max_steps => MAX_STEPS
            ),
            Dict(
                :map => map,
                :start_position => (366.0, 173.0),
                :start_angle => 315.0,
                :max_steps => MAX_STEPS
            )
        ],
        :changeable_validation_kwargs_list => [
            Dict(
                :map => map,
                :start_position => (504.0, 744.0),
                :start_angle => 0.0,
                :max_steps => 5000
            )
        ]
    ),
    :neural_network => Dict(
        :name => :MLP_NN,
        :kwargs => Dict(
            :input_size => 10,
            :output_size => 6,
            :hidden_layers => 2,
            :hidden_neurons => 256,  # 64
            :dropout => 0.0,
            :activation_function => :relu,  # :relu
            :last_activation_function => [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
        )
    ),
    :Genetic_Algorithm => Dict(
        :population => 100,
        :max_evaluations => 100000,
        :epochs => 10000,
        :mutation_factor => 0.1,
        :crosses_per_epoch => 99,
        :save_logs_every_n_epochs => 10,
        :max_threads => 0,
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs"
    ),
    :Differential_Evolution => Dict(
        :population => 2000,
        :max_evaluations => 100000,
        :epochs => 10000,
        :cross_prob => 0.9,
        :diff_weight => 0.8,
        :save_logs_every_n_epochs => 50,
        :max_threads => 0,
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs"
    ),
    :Evolutionary_Mutate_Population => Dict(
        :population_size => 100,
        # :max_evaluations => 100000,
        # :max_generations => 1000,
        :mutation_rate => 0.1,
        # :mutation_controller => Dict(
        #     :name => :Mut_One,
        #     :kwargs => Dict(
        #         :mutation_factor => 0.1,
        #         :use_children => false
        #     )
        # ),
        :n_threads => 8,
        # :save_logs_every_n_epochs => 50,
        # :logs_path => raw"logs"
    ),
    :Evolutionary_Mutate_Population_Original => Dict(
        :population => 5000,
        :best_base_N => 100,
        :max_evaluations => 100000,
        :max_generations => 1000,
        :mutation_controller => Dict(
            :name => :Mut_One,
            :kwargs => Dict(
                :mutation_factor => 0.1,
                :use_children => false
            )
        ),
        :n_threads => 8,
        :save_logs_every_n_epochs => 50,
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs"
    ),
    :GESMR => Dict(
        :population => 100,
        :epochs => 400,
        :k_groups => 10,
        :mut_range => (0.08, 0.1),
        :individual_ratio_breed => 0.5,
        :mutation_ratio_breed => 0.5,
        :mutation_ratio_mutate => 0.5,
        :max_threads => 8,
        :save_logs_every_n_epochs => 5,
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs"
    ),
    :Param_Les_Ev_Mut_Pop => Dict(
        :epochs => 10000,
        :mutation_controller => Dict(
            :name => :Mut_Prob,
            :kwargs => Dict(
                :mem_size => 10,
                :initial_mut_fact_range => (0.05, 0.15),  # (0.001, 0.2)
                :survival_rate => 0.01,
                :learning_rate => 0.1
            )
        ),
        :max_threads => 22,
        :save_logs_every_n_epochs => 300,
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs"
    ),
    :Evolutionary_Strategy => Dict(
        :permutations => 1000,
        :max_evaluations => 100000,
        :epochs => 10000,
        :sigma_change => 0.01,
        :learning_rate => 0.1,
        :save_logs_every_n_epochs => 20,
        :max_threads => 0,
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs"
    ),
)
