using Images
using FileIO
import Flux

map_image_path = raw"data\map2.png"
car_image_path = raw"data\car.png"
map_image_path_2 = raw"data\map.png"

img = Gray.(load(map_image_path))
map = BitArray(Array(img) .> 0.5)

img2 = Gray.(load(map_image_path_2))
map_2 = BitArray(Array(img2) .> 0.5)

MAX_STEPS = 2000

CONSTANTS_DICT = Dict(
    # ------------------------------------------------------------------------------------
    # Universal staff
    :run_config => Dict(
        :max_generations => 1000,
        :max_evaluations => 1_000_000,
        :log => true,
        :visualize_each_n_epochs => 0,
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


    # ------------------------------------------------------------------------------------
    # method specific staff
    :ContinuousStatesGroupingSimpleGA => Dict(
        :env_wrapper => Dict(
            :encoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 10,
                    :output_size => 16,  # 16
                    :hidden_layers => 2,
                    :hidden_neurons => 32,  # 32
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,
                    :last_activation_function => :none
                )
            ),
            :decoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 16,  # 16
                    :output_size => 10,  # it should be 10, 9 is for normal learning
                    :hidden_layers => 2,  # was 1
                    :hidden_neurons => 32,  # 64
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :relu,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => Flux.mse  # was Flux.mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # IDK why, but in early tests clearly 0.0 was the best, so MMD wasnt used at all, there was a huge difference
                :learning_rate => 0.001
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :n_clusters => 40,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :hclust_distance => :complete,  # :ward or :single or :complete or :average
            :hclust_time => :complete,  # :ward or :single or :complete or :average
            :m_value => 2,  # 2 is better than 1
            :exemplars_clustering => :pam  # :genie or :kmedoids or :pam
        ),
        :individuals_n => 30,
        :fihc => Dict(
            :fihc_mode => :fihc_cont,
            :norm_mode => :d_sum,
            :factor => 0.5,
            :genes_combination => :hier,
            :random_matrix_mode => :randn
        ),
    ),


    :ContinuousStatesGroupingP3 => Dict(
        :env_wrapper => Dict(
            :encoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 10,
                    :output_size => 16,  # 16
                    :hidden_layers => 2,
                    :hidden_neurons => 32,  # 32
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,
                    :last_activation_function => :none
                )
            ),
            :decoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 16,  # 16
                    :output_size => 10,  # it should be 10, 9 is for normal learning
                    :hidden_layers => 2,  # was 1
                    :hidden_neurons => 32,  # 64
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :relu,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => Flux.mse  # was Flux.mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # IDK why, but in early tests clearly 0.0 was the best, so MMD wasnt used at all, there was a huge difference
                :learning_rate => 0.001
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :n_clusters => 40,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :hclust_distance => :ward,  # :ward or :single or :complete or :average
            :hclust_time => :ward,  # :ward or :single or :complete or :average
            :m_value => 2,  # 2 is better than 1
            :exemplars_clustering => :pam  # :genie or :kmedoids or :pam
        )
    ),









    :StatesGroupingGA => Dict(
        :env_wrapper => Dict(
            :encoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 10,
                    :output_size => 16,  # 16
                    :hidden_layers => 2,
                    :hidden_neurons => 32,  # 32
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :last_activation_function => :none
                )
            ),
            :decoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 16,  # 16
                    :output_size => 10,  # it should be 10, 9 is for normal learning
                    :hidden_layers => 2,  # was 1
                    :hidden_neurons => 32,  # 64
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :relu,
                    :last_activation_function => :none, # was :none
                    :loss => Flux.mse  # was Flux.mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # IDK why, but in early tests clearly 0.0 was the best, so MMD wasnt used at all, there was a huge difference
                :learning_rate => 0.001
            ),
            :game_decoder_dict => Dict(  # used only in the first collection on states
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 16,  # 16
                    :output_size => 9,
                    :hidden_layers => 2,
                    :hidden_neurons => 32,  # 64
                    :dropout => 0.0,  # 0.5,
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,  # :relu,
                    :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                    :loss => Flux.crossentropy
                )
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :fuzzy_logic_of_n_closest => -1,  # it should be switched off, so set it e.g. to -1
            :m_value => 2,  # should test 1 vs 2
            :n_clusters => 40,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :hclust_distance => :complete,  # :ward or :single or :complete or :average
            :hclust_time => :single,  # :ward or :single or :complete or :average
            :exemplars_clustering => :pam  # :genie or :kmedoids or :pam
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
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
        :neural_network => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 10,
                :output_size => 9,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 64,  # 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => Flux.kldivergence,
            )
        ),
    ),

    :Differential_Evolution => Dict(
        :population => 2000,
        :max_evaluations => 100000,
        :epochs => 10000,
        :cross_prob => 0.9,
        :diff_weight => 0.8,
        :save_logs_every_n_epochs => 50,
        :max_threads => 0,
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
        :neural_network => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 10,
                :output_size => 9,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 64,  # 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => Flux.kldivergence
            )
        ),
    ),

    :Evolutionary_Mutate_Population => Dict(
        :population_size => 100, # 400
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
        :neural_network => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 10,
                :output_size => 9,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 64,  # 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => Flux.kldivergence
            )
        ),
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
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
        :neural_network => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 10,
                :output_size => 9,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 64,  # 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => Flux.kldivergence
            )
        ),
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
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
        :neural_network => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 10,
                :output_size => 9,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 64,  # 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => Flux.kldivergence
            )
        ),
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
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
        :neural_network => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 10,
                :output_size => 9,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 64,  # 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => Flux.kldivergence
            )
        ),
    ),
    
    :Evolutionary_Strategy => Dict(
        :permutations => 1000,
        :max_evaluations => 100000,
        :epochs => 10000,
        :sigma_change => 0.01,
        :learning_rate => 0.1,
        :save_logs_every_n_epochs => 20,
        :max_threads => 0,
        :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
        :neural_network => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 10,
                :output_size => 9,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 64,  # 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => Flux.kldivergence
            )
        ),
    ),
)
