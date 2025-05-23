using Images
using FileIO

# Important! types inside the dictionary should be from Julia std, simple types if possible, so e.g.
# Dict, Symbol, Int, Float, Vector, BitArray
# Shouldnt use e.g. things from Flux etc.

map_image_path = raw"data\map2.png"
car_image_path = raw"data\car.png"
map_image_path_2 = raw"data\map.png"

img = Gray.(load(map_image_path))
map = BitArray(Array(img) .> 0.5)

img2 = Gray.(load(map_image_path_2))
map_2 = BitArray(Array(img2) .> 0.5)


CARS_DICT = Dict(
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
            :max_steps => 2000,
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
            ),
            Dict(
                :map => map,
                :start_position => (504.0, 744.0),
                :start_angle => 180.0,
            ),
            Dict(
                :map => map,
                :start_position => (425.0, 337.0),
                :start_angle => 170.0,
            ),
            Dict(
                :map => map,
                :start_position => (283.0, 536.0),
                :start_angle => 200.0,
            ),
            Dict(
                :map => map,
                :start_position => (665.0, 400.0),
                :start_angle => 270.0,
            ),
            Dict(
                :map => map,
                :start_position => (366.0, 173.0),
                :start_angle => 315.0,
            )
        ],
        :changeable_validation_kwargs => 
            Dict(
                :map => map,
                :start_position => (504.0, 744.0),
                :start_angle => 0.0,
                :max_steps => 5000
            )
    ),


    # ------------------------------------------------------------------------------------
    # method specific staff
    :EncoderOutputES => Dict(
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
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 20_000,
            :environment_norm => nothing,
            :activation_function => :none,
            :verbose => false,
        ),
        # :new_es_after_n_iterations => 10,
        # :es_type => :CMAES,
        # :es_kwargs => Dict(
        #     :sigma => 0.3f0,
        #     :lambda_n => 500,  # it might be left empty
        # ),
        # :es_type => :DE,
        # :es_kwargs => Dict(
        #     :base_mode => :best, # :best or :rand
        #     # :sigma => 0.3f0, # here it doesnt really matter
        #     :cross_per_symbol => :all, # :row or :col or :all
        #     :parents_n => 1000,
        #     :f_param => 0.7f0,
        #     :cross_param => 0.5f0,
        #     :norm_std => false,
        # ),
        :es_type => :ARS,
        :es_kwargs => Dict(
            :sigma => 0.03f0, # here it doesnt really matter
            :step_size => 0.03f0,
            :child_n => 1000,
            :best_n => 500,
        ),
        :individual_config => Dict(
            :fitnesses_reduction => :sum,
        ),
    ),

    :ContinuousStatesGroupingES => Dict(
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
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :environment_norm => nothing,
            :n_clusters => 20,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :exemplars_clustering => :my_pam_random_increasing,  # :genie or :kmedoids or :pam
            :exemplar_nn=>Dict(
                :interaction_method=>:cosine,
                :membership_normalization=>:mval_2,  # can be either mval_2 for 2 Int type or mval_1_5 for 1.5 Float32
                :activation_function=>:none,  # for car racing should be :none, for humanoid should be :tanh
            ),
        ),
        :new_es_after_n_iterations => 100,
        :es_type => :CMAES,
        :es_kwargs => Dict(
            :sigma => 1f0,
            :lambda_n => 1000,  # it might be left empty
        ),
        :individual_config => Dict(
            :norm_genes => :none,  # make sure initial settings will not destroy anything, it is not used 
            :levels_mode => :flat,  # make sure initial settings will not destroy anything, it is not used, make sure I do not doo additional calculations
            :fitnesses_reduction => :mean,
        ),
    ),





    :ContinuousStatesGroupingFIHC => Dict(
        :env_wrapper => Dict(
            # for car racing:
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
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :n_clusters => 20,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :exemplars_clustering => :my_pam_random_increasing,  # :genie or :kmedoids or :pam
            :environment_norm => nothing,
            :exemplar_nn=>Dict(
                :interaction_method=>:cosine,
                :membership_normalization=>:mval_2,  # can be either mval_2 for 2 Int type or mval_1_5 for 1.5 Float32
                :activation_function=>:none,  # for car racing should be :none, for humanoid should be :tanh
            ),
        ),
        :new_individual_after_n_no_improvements => 10,
        :new_individual_genes => :best,  # :rand or :best
        :individual_config => Dict(
            :initial_genes_mode => :std,
            :norm_genes => :none,  # for car racing should be std, for humanoid should be :none
            :levels_mode => :time_markov,  # all, flat, time_markov, time_mine, latent
            :levels_hclust => :complete,  # :ward or :single or :complete or :average
            :levels_construct_mode => :equal_up,  # equal_up, equal_down, priority_up, priority_down
            :fitnesses_reduction => :median,
            # fihc things
            :fihc_f => 0.1,
            :fihc_n_times => 5,
            :fihc_same_per_gene => true,
        ),
    ),


    :ContinuousStatesGroupingDE => Dict(
        :env_wrapper => Dict(
            # for car racing:
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
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :n_clusters => 20,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :exemplars_clustering => :my_pam_random_increasing,  # :genie or :kmedoids or :pam
            :environment_norm => nothing,
            :exemplar_nn=>Dict(
                :interaction_method=>:cosine,
                :membership_normalization=>:mval_2,  # can be either mval_2 for 2 Int type or mval_1_5 for 1.5 Float32
                :activation_function=>:none,  # for car racing should be :none, for humanoid should be :tanh
            ),
        ),
        :individuals_n => 50,
        :new_individual_each_n_epochs => 1,
        :new_individual_genes => :rand,  # :rand or :best
        :individual_config => Dict(
            :initial_genes_mode => :std,
            :norm_genes => :std,  # for car racing should be std, for humanoid should be :none
            :levels_mode => :time_markov,  # all, flat, time_markov, time_mine, latent
            :levels_hclust => :complete,  # :ward or :single or :complete or :average
            :levels_construct_mode => :equal_up,  # equal_up, equal_down, priority_up, priority_down
            :base_mode => :best, # :best or :rand or :self
            :mask_mode => :per_gene,  # :per_gene or :per_value
            :cross_n_times => 1,  # how many times to cross genes per one generation
            :cross_f => 0.8,
            :cross_prob => 1.0,
            :fitnesses_reduction => :sum,
        ),
    ),

    :Evolutionary_Mutate_Population => Dict(
        :population_size => 100, # 400
        :mutation_rate => 0.05,
        :environment_norm => nothing,
        :neural_network_data => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 10,
                :output_size => 9,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 64,  # was 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => :kldivergence
            )
        ),
    ),
)



HUMANOID_DICT = Dict(
    # ------------------------------------------------------------------------------------
    # Universal staff
    :run_config => Dict(
        :max_generations => 1000,
        :max_evaluations => 1_000_000,
        :log => true,
        :visualize_each_n_epochs => 0,
    ),
    :environment => Dict(
        :name => :Humanoid,
        :visualization => Dict(
        ),
        :universal_kwargs => Dict(
            :base_version => :humanoid_v5,
            :max_steps => 1000,
            :reset_noise_scale => 0.01,
            :reset_random_generator => true,
            :additional_normalization => true,
        ),
        :changeable_training_kwargs_list => [
            Dict(
                :seed => 0,
            ),
            Dict(
                :seed => 1,
            ),
            Dict(
                :seed => 2,
            ),
            Dict(
                :seed => 3,
            ),
            Dict(
                :seed => 4,
            ),
            Dict(
                :seed => 5,
            )
        ],
        :changeable_validation_kwargs => Dict()
        
    ),


    # ------------------------------------------------------------------------------------
    # method specific staff

    :EncoderOutputES => Dict(
        :env_wrapper => Dict(
            :encoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 348,
                    :output_size => 348,  # 16
                    :hidden_layers => 0,
                    :hidden_neurons => 256,  # 32
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,
                    :last_activation_function => :none
                )
            ),
            :decoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 348,  # 16
                    :output_size => 348,  # it should be 10, 9 is for normal learning
                    :hidden_layers => 0,  # was 1
                    :hidden_neurons => 256,  # 64
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            # :encoder_dict => Dict(
            #     :name => :MLP_NN,
            #     :kwargs => Dict(
            #         :input_size => 348,
            #         :output_size => 16,  # 16
            #         :hidden_layers => 2,
            #         :hidden_neurons => 256,  # 32
            #         :dropout => 0.0,  # 0.5
            #         :activation_function => :relu,  # :relu
            #         :input_activation_function => :none,
            #         :last_activation_function => :none
            #     )
            # ),
            # :decoder_dict => Dict(
            #     :name => :MLP_NN,
            #     :kwargs => Dict(
            #         :input_size => 16,  # 16
            #         :output_size => 348,  # it should be 10, 9 is for normal learning
            #         :hidden_layers => 2,  # was 1
            #         :hidden_neurons => 256,  # 64
            #         :dropout => 0.0,  # 0.5
            #         :activation_function => :relu,  # :relu
            #         :input_activation_function => :none,  # shouldnt it be :none?
            #         :last_activation_function => :none, # was :none
            #         :loss => :mse
            #     )
            # ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 20_000,
            :environment_norm => :NormMatrixASSEQ,
            :activation_function => :none,
            :verbose => false,
        ),
        # :new_es_after_n_iterations => 10,
        # :es_type => :CMAES,
        # :es_kwargs => Dict(
        #     :sigma => 0.3f0,
        #     :lambda_n => 500,  # it might be left empty
        # ),
        # :es_type => :DE,
        # :es_kwargs => Dict(
        #     :base_mode => :best, # :best or :rand
        #     # :sigma => 0.3f0, # here it doesnt really matter
        #     :cross_per_symbol => :all, # :row or :col or :all
        #     :parents_n => 1000,
        #     :f_param => 0.7f0,
        #     :cross_param => 0.5f0,
        #     :norm_std => false,
        # ),
        :es_type => :ARS,
        :es_kwargs => Dict(
            :sigma => 0.007f0, # here it doesnt really matter
            :step_size => 0.02f0,
            :child_n => 230,
            # :best_n => 230,
        ),
        :individual_config => Dict(
            :fitnesses_reduction => :mean,
        ),
    ),



    :ContinuousStatesGroupingES => Dict(
        :env_wrapper => Dict(
            :encoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 376,
                    :output_size => 16,  # 16
                    :hidden_layers => 2,
                    :hidden_neurons => 256,  # 32
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
                    :output_size => 376,  # it should be 10, 9 is for normal learning
                    :hidden_layers => 2,  # was 1
                    :hidden_neurons => 256,  # 64
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 20_000,
            :environment_norm => :NormMatrixASSEQ,
            :n_clusters => 40,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :exemplars_clustering => :my_pam_random_increasing,  # :genie or :kmedoids or :pam
            :exemplar_nn=>Dict(
                :interaction_method=>:cosine,
                :membership_normalization=>:mval_2,  # can be either mval_2 for 2 Int type or mval_1_5 for 1.5 Float32
                :activation_function=>:tanh,  # for car racing should be :none, for humanoid should be :tanh
            ),
        ),
        :new_es_after_n_iterations => 10,
        :es_type => :CMAES,
        :es_kwargs => Dict(
            :sigma => 0.01f0,
            :lambda_n => 100,  # it might be left empty
        ),
        :individual_config => Dict(
            :norm_genes => :none,  # make sure initial settings will not destroy anything, it is not used 
            :levels_mode => :flat,  # make sure initial settings will not destroy anything, it is not used, make sure I do not doo additional calculations
            :fitnesses_reduction => :mean,
        ),
    ),

    :ContinuousStatesGroupingFIHC => Dict(
        :env_wrapper => Dict(
            :encoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 376,
                    :output_size => 32,  # 16
                    :hidden_layers => 2,
                    :hidden_neurons => 256,  # 32
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,
                    :last_activation_function => :none
                )
            ),
            :decoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 32,  # 16
                    :output_size => 376,  # it should be 10, 9 is for normal learning
                    :hidden_layers => 2,  # was 1
                    :hidden_neurons => 256,  # 64
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :environment_norm => :NormMatrixASSEQ,
            :n_clusters => 15,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :exemplars_clustering => :my_pam_random_increasing,  # :genie or :kmedoids or :pam
            :exemplar_nn=>Dict(
                :interaction_method=>:cosine,
                :membership_normalization=>:mval_2,  # can be either mval_2 for 2 Int type or mval_1_5 for 1.5 Float32
                :activation_function=>:tanh,  # for car racing should be :none, for humanoid should be :tanh
            ),
        ),
        :new_individual_after_n_no_improvements => 3,
        :new_individual_genes => :rand,  # :rand or :best
        :individual_config => Dict(
            :initial_genes_mode => :std,
            :norm_genes => :tanh,  # for car racing should be std, for humanoid should be :none
            :levels_mode => :time_markov,  # all, flat, time_markov, time_mine, latent
            :levels_hclust => :complete,  # :ward or :single or :complete or :average
            :levels_construct_mode => :priority_up,  # equal_up, equal_down, priority_up, priority_down
            :fitnesses_reduction => :median,

            :per_column_norm => false,

            # fihc things
            :fihc_f => 0.5,
            :fihc_n_times => 5,
            :fihc_same_per_gene => true,
        ),
    ),


    :ContinuousStatesGroupingDE => Dict(
        :env_wrapper => Dict(
            :encoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 376,
                    :output_size => 32,  # 16
                    :hidden_layers => 2,
                    :hidden_neurons => 256,  # 32
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,
                    :last_activation_function => :none
                )
            ),
            :decoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 32,  # 16
                    :output_size => 376,  # it should be 10, 9 is for normal learning
                    :hidden_layers => 2,  # was 1
                    :hidden_neurons => 256,  # 64
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :environment_norm => :NormMatrixASSEQ,
            :n_clusters => 20,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :exemplars_clustering => :my_pam_random_increasing,  # :genie or :kmedoids or :pam
            :exemplar_nn=>Dict(
                :interaction_method=>:cosine,
                :membership_normalization=>:mval_2,  # can be either mval_2 for 2 Int type or mval_1_5 for 1.5 Float32
                :activation_function=>:tanh,  # for car racing should be :none, for humanoid should be :tanh
            ),
        ),
        :individuals_n => 50,
        :new_individual_each_n_epochs => 1,
        :new_individual_genes => :rand,  # :rand or :best
        :individual_config => Dict(
            :initial_genes_mode => :std,
            :norm_genes => :none,  # for car racing should be std, for humanoid should be :none
            :levels_mode => :time_markov,  # all, flat, time_markov, time_mine, latent
            :levels_hclust => :complete,  # :ward or :single or :complete or :average
            :levels_construct_mode => :priority_up,  # equal_up, equal_down, priority_up, priority_down
            :base_mode => :rand, # :best or :rand or :self
            :mask_mode => :per_gene,  # :per_gene or :per_value

            :per_column_norm => true,

            :cross_n_times => 1,  # how many times to cross genes per one generation
            :cross_f => 0.8,
            :cross_prob => 1.0,
            :fitnesses_reduction => :median,
        ),
    ),

    :Evolutionary_Mutate_Population => Dict(
        :population_size => 100, # 400
        :mutation_rate => 0.05,
        :environment_norm => :NormMatrixASSEQ,
        :environment_norm_individuals => 30,
        :environment_norm_activation => :tanh,
        :fitnesses_reduction_method => :median,
        :neural_network_data => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 376,
                :output_size => 17,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 256,  # was 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :tanh,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => :kldivergence
            )
        ),
    ),
)


HALF_CHEETAH_DICT = Dict(
    # ------------------------------------------------------------------------------------
    # Universal staff
    :run_config => Dict(
        :max_generations => 1000,
        :max_evaluations => 1_000_000,
        :log => true,
        :visualize_each_n_epochs => 0,
    ),
    :environment => Dict(
        :name => :HalfCheetah,
        :visualization => Dict(
        ),
        :universal_kwargs => Dict(
            :base_version => :half_cheetah_v5,
            :max_steps => 1000,
            :reset_noise_scale => 0.1,
            :reset_random_generator => true,
        ),
        :changeable_training_kwargs_list => [
            Dict(
                :seed => 0,
            ),
            Dict(
                :seed => 1,
            ),
            Dict(
                :seed => 2,
            ),
            Dict(
                :seed => 3,
            ),
            Dict(
                :seed => 4,
            ),
            Dict(
                :seed => 5,
            )
        ],
        :changeable_validation_kwargs => Dict()
        
    ),


    # ------------------------------------------------------------------------------------
    # method specific staff

        :ContinuousStatesGroupingFIHC => Dict(
        :env_wrapper => Dict(
            :encoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 376,
                    :output_size => 32,  # 16
                    :hidden_layers => 2,
                    :hidden_neurons => 256,  # 32
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,
                    :last_activation_function => :none
                )
            ),
            :decoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 32,  # 16
                    :output_size => 376,  # it should be 10, 9 is for normal learning
                    :hidden_layers => 2,  # was 1
                    :hidden_neurons => 256,  # 64
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :environment_norm => :NormMatrixASSEQ,
            :n_clusters => 20,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :exemplars_clustering => :my_pam_random_increasing,  # :genie or :kmedoids or :pam
            :exemplar_nn=>Dict(
                :interaction_method=>:cosine,
                :membership_normalization=>:mval_2,  # can be either mval_2 for 2 Int type or mval_1_5 for 1.5 Float32
                :activation_function=>:tanh,  # for car racing should be :none, for humanoid should be :tanh
            ),
        ),
        :new_individual_after_n_no_improvements => 10,
        :new_individual_genes => :best,  # :rand or :best
        :individual_config => Dict(
            :initial_genes_mode => :std,
            :norm_genes => :none,  # for car racing should be std, for humanoid should be :none
            :levels_mode => :time_markov,  # all, flat, time_markov, time_mine, latent
            :levels_hclust => :complete,  # :ward or :single or :complete or :average
            :levels_construct_mode => :equal_up,  # equal_up, equal_down, priority_up, priority_down
            :fitnesses_reduction => :median,
            # fihc things
            :fihc_f => 0.1,
            :fihc_n_times => 5,
            :fihc_same_per_gene => true,
        ),
    ),


    :ContinuousStatesGroupingDE => Dict(
        :env_wrapper => Dict(
            :encoder_dict => Dict(
                :name => :MLP_NN,
                :kwargs => Dict(
                    :input_size => 17,
                    :output_size => 16,  # 16
                    :hidden_layers => 2,
                    :hidden_neurons => 64,  # 32
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
                    :output_size => 17,  # it should be 10, 9 is for normal learning
                    :hidden_layers => 2,  # was 1
                    :hidden_neurons => 64,  # 64
                    :dropout => 0.0,  # 0.5
                    :activation_function => :relu,  # :relu
                    :input_activation_function => :none,  # shouldnt it be :none?
                    :last_activation_function => :none, # was :none
                    :loss => :mse
                )
            ),
            :autoencoder_dict => Dict(
                :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
                :learning_rate => 0.001,  # 0.001
                :weight_decay => 0.0
            ),
            :initial_space_explorers_n => 30,
            :max_states_considered => 10_000,
            :environment_norm => :NormMatrixASSEQ,
            :n_clusters => 20,  # 40 and 200 works very well, should try different values
            :verbose => false,
            :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
            :exemplars_clustering => :my_pam_random_increasing,  # :genie or :kmedoids or :pam
            :exemplar_nn=>Dict(
                :interaction_method=>:cosine,
                :membership_normalization=>:mval_2,  # can be either mval_2 for 2 Int type or mval_1_5 for 1.5 Float32
                :activation_function=>:tanh,  # for car racing should be :none, for humanoid should be :tanh
            ),
        ),
        :individuals_n => 50,
        :new_individual_each_n_epochs => 1,
        :new_individual_genes => :rand,  # :rand or :best
        :individual_config => Dict(
            :initial_genes_mode => :std,
            :norm_genes => :none,  # for car racing should be std, for humanoid should be :none
            :levels_mode => :time_markov,  # all, flat, time_markov, time_mine, latent
            :levels_hclust => :complete,  # :ward or :single or :complete or :average
            :levels_construct_mode => :equal_up,  # equal_up, equal_down, priority_up, priority_down
            :base_mode => :best, # :best or :rand or :self
            :mask_mode => :per_gene,  # :per_gene or :per_value
            :cross_n_times => 1,  # how many times to cross genes per one generation
            :cross_f => 0.8,
            :cross_prob => 1.0,
            :fitnesses_reduction => :median,
        ),
    ),

    :Evolutionary_Mutate_Population => Dict(
        :population_size => 100, # 400
        :mutation_rate => 0.05,
        :environment_norm => :NormMatrixASSEQ,
        :environment_norm_individuals => 30,
        :environment_norm_activation => :tanh,
        :fitnesses_reduction_method => :median,
        :neural_network_data => Dict(
            :name => :MLP_NN,
            :kwargs => Dict(
                :input_size => 17,
                :output_size => 6,  # 6 # 3 # 9
                :hidden_layers => 2,
                :hidden_neurons => 64,  # was 64
                :dropout => 0.0,
                :activation_function => :relu,  # :relu
                :last_activation_function => :tanh,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
                :loss => :kldivergence
            )
        ),
    ),
)







# CONSTANTS_DICT = CARS_DICT
# CONSTANTS_DICT = HALF_CHEETAH_DICT
CONSTANTS_DICT = HUMANOID_DICT






    # -------------------------------------------------------------------------------------
    # Old, depracated methods
    # -------------------------------------------------------------------------------------

#     :ContinuousStatesGroupingP3 => Dict(
#         :env_wrapper => Dict(
#             :encoder_dict => Dict(
#                 :name => :MLP_NN,
#                 :kwargs => Dict(
#                     :input_size => 10,
#                     :output_size => 16,  # 16
#                     :hidden_layers => 2,
#                     :hidden_neurons => 32,  # 32
#                     :dropout => 0.0,  # 0.5
#                     :activation_function => :relu,  # :relu
#                     :input_activation_function => :none,
#                     :last_activation_function => :none
#                 )
#             ),
#             :decoder_dict => Dict(
#                 :name => :MLP_NN,
#                 :kwargs => Dict(
#                     :input_size => 16,  # 16
#                     :output_size => 10,  # it should be 10, 9 is for normal learning
#                     :hidden_layers => 2,  # was 1
#                     :hidden_neurons => 32,  # 64
#                     :dropout => 0.0,  # 0.5
#                     :activation_function => :relu,  # :relu
#                     :input_activation_function => :relu,  # shouldnt it be :none?
#                     :last_activation_function => :none, # was :none
#                     :loss => :mse
#                 )
#             ),
#             :autoencoder_dict => Dict(
#                 :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
#                 :learning_rate => 0.001
#             ),
#             :initial_space_explorers_n => 30,
#             :max_states_considered => 10_000,
#             :n_clusters => 40,  # 40 and 200 works very well, should try different values
#             :verbose => true,
#             :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
#             :hclust_distance => :complete,  # :ward or :single or :complete or :average
#             :hclust_time => :complete,  # :ward or :single or :complete or :average
#             :exemplars_clustering => :pam,  # :genie or :kmedoids or :pam
#             :distance_membership_levels_method => :hclust_complete,
#             :time_distance_tree => :markov,  # :markov or :mine
#             :exemplar_nn=>Dict(
#                 :interaction_method=>:cosine,
#                 :membership_normalization=>:mval_2,
#                 :activation_function=>:dsum,
#             ),
#         ),
#         :fihc => Dict(
#             :fihc_mode => :per_gene_rand,
#             :norm_mode => :d_sum,
#             :factor => 1.0,
#             :hier_factor => 1.0,
#             :random_matrix_mode => :rand_n_different,
#             :local_fuzzy => :none,
#         ),
#         :cross => Dict(
#             :norm_mode => :d_sum,
#             :self_vs_other => (0.0, 1.0),
#             :genes_combinations => :tree_up, # :tree_up or :tree_down or :flat or :all
#             :strategy => :all_comb,  # :one_rand or :one_tournament or or :all_seq or :all_comb or rand_comb
#         ),
#         :initial_genes_mode => :scale,  # :scale or :softmax
#     ),






#     :ContinuousStatesGroupingSimpleGA => Dict(
#         :env_wrapper => Dict(
#             :encoder_dict => Dict(
#                 :name => :MLP_NN,
#                 :kwargs => Dict(
#                     :input_size => 10,
#                     :output_size => 16,  # 16
#                     :hidden_layers => 2,
#                     :hidden_neurons => 32,  # 32
#                     :dropout => 0.0,  # 0.5
#                     :activation_function => :relu,  # :relu
#                     :input_activation_function => :none,
#                     :last_activation_function => :none
#                 )
#             ),
#             :decoder_dict => Dict(
#                 :name => :MLP_NN,
#                 :kwargs => Dict(
#                     :input_size => 16,  # 16
#                     :output_size => 10,  # it should be 10, 9 is for normal learning
#                     :hidden_layers => 2,  # was 1
#                     :hidden_neurons => 32,  # 64
#                     :dropout => 0.0,  # 0.5
#                     :activation_function => :relu,  # :relu
#                     :input_activation_function => :none,  # shouldnt it be :none?
#                     :last_activation_function => :none, # was :none
#                     :loss => :mse
#                 )
#             ),
#             :autoencoder_dict => Dict(
#                 :mmd_weight => 0.0,  # turns out it might be beneficial to set it to 0.01, so maybe in the future compare e.g. 0.0, 0.01, 0.1
#                 :learning_rate => 0.001,  # 0.001
#                 :weight_decay => 0.0
#             ),
#             :initial_space_explorers_n => 30,
#             :max_states_considered => 10_000,
#             :n_clusters => 20,  # 40 and 200 works very well, should try different values
#             :verbose => false,
#             :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
#             :hclust_distance => :complete,  # :ward or :single or :complete or :average
#             :hclust_time => :complete,  # :ward or :single or :complete or :average
#             :exemplars_clustering => :my_pam_random_increasing,  # :genie or :kmedoids or :pam
#             :distance_membership_levels_method => :hclust_complete,
#             :time_distance_tree => :markov,  # :markov or :mine
#             :exemplar_nn=>Dict(
#                 :interaction_method=>:cosine,
#                 :membership_normalization=>:mval_2,
#                 :activation_function=>:none,
#             ),
#         ),
#         :individuals_n => 50,
#         :initial_genes_mode => :std,  # :scale or :softmax
#         :new_individual_each_n_epochs => 1,
#         :new_individual_genes => :rand,
#         :fihc => Dict(
#             :fihc_mode => :none,
#             :levels_mode => :distance,  # mostly :distance or :time_down
#             :norm_mode => :std,
#             :factor => 1.0,
#             :hier_factor => 1.0,
#             :random_matrix_mode => :rand_n_different,
#             :local_fuzzy => :none,
#         ),
#         :cross => Dict(
#             :norm_mode => :std,
#             :self_vs_other => (0.0, 1.0),
#             :genes_combinations => :tree_up, # :tree_up or :tree_down or :flat or :all
#             :strategy => :yes,
#             :cross_strategy => :best,
#             :cross_prob => 1.0,
#             :f_value => 0.8,
#             :cross_prob_mode => :column,
#         ),
        
#     ),









#     :StatesGroupingGA => Dict(
#         :env_wrapper => Dict(
#             :encoder_dict => Dict(
#                 :name => :MLP_NN,
#                 :kwargs => Dict(
#                     :input_size => 10,
#                     :output_size => 16,  # 16
#                     :hidden_layers => 2,
#                     :hidden_neurons => 32,  # 32
#                     :dropout => 0.0,  # 0.5
#                     :activation_function => :relu,  # :relu
#                     :last_activation_function => :none
#                 )
#             ),
#             :decoder_dict => Dict(
#                 :name => :MLP_NN,
#                 :kwargs => Dict(
#                     :input_size => 16,  # 16
#                     :output_size => 10,  # it should be 10, 9 is for normal learning
#                     :hidden_layers => 2,  # was 1
#                     :hidden_neurons => 32,  # 64
#                     :dropout => 0.0,  # 0.5
#                     :activation_function => :relu,  # :relu
#                     :input_activation_function => :relu,
#                     :last_activation_function => :none, # was :none
#                     :loss => :mse
#                 )
#             ),
#             :autoencoder_dict => Dict(
#                 :mmd_weight => 0.0,  # IDK why, but in early tests clearly 0.0 was the best, so MMD wasnt used at all, there was a huge difference
#                 :learning_rate => 0.001
#             ),
#             :game_decoder_dict => Dict(  # used only in the first collection on states
#                 :name => :MLP_NN,
#                 :kwargs => Dict(
#                     :input_size => 16,  # 16
#                     :output_size => 9,
#                     :hidden_layers => 2,
#                     :hidden_neurons => 32,  # 64
#                     :dropout => 0.0,  # 0.5,
#                     :activation_function => :relu,  # :relu
#                     :input_activation_function => :none,  # :relu,
#                     :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
#                     :loss => :crossentropy
#                 )
#             ),
#             :initial_space_explorers_n => 30,
#             :max_states_considered => 10_000,
#             :fuzzy_logic_of_n_closest => -1,  # it should be switched off, so set it e.g. to -1
#             :m_value => 2,  # should test 1 vs 2
#             :n_clusters => 40,  # 40 and 200 works very well, should try different values
#             :verbose => false,
#             :distance_metric => :cosine,  # :euclidean or :cosine or :cityblock, after some initial tests it should definatelly be cosine!
#             :hclust_distance => :complete,  # :ward or :single or :complete or :average
#             :hclust_time => :single,  # :ward or :single or :complete or :average
#             :exemplars_clustering => :pam  # :genie or :kmedoids or :pam
#         )
#     ),









#     :Genetic_Algorithm => Dict(
#         :population => 100,
#         :max_evaluations => 100000,
#         :epochs => 10000,
#         :mutation_factor => 0.1,
#         :crosses_per_epoch => 99,
#         :save_logs_every_n_epochs => 10,
#         :max_threads => 0,
#         :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
#         :neural_network => Dict(
#             :name => :MLP_NN,
#             :kwargs => Dict(
#                 :input_size => 10,
#                 :output_size => 9,  # 6 # 3 # 9
#                 :hidden_layers => 2,
#                 :hidden_neurons => 64,  # 64
#                 :dropout => 0.0,
#                 :activation_function => :relu,  # :relu
#                 :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
#                 :loss => :kldivergence,
#             )
#         ),
#     ),

#     :Differential_Evolution => Dict(
#         :population => 2000,
#         :max_evaluations => 100000,
#         :epochs => 10000,
#         :cross_prob => 0.9,
#         :diff_weight => 0.8,
#         :save_logs_every_n_epochs => 50,
#         :max_threads => 0,
#         :logs_path => raw"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
#         :neural_network => Dict(
#             :name => :MLP_NN,
#             :kwargs => Dict(
#                 :input_size => 10,
#                 :output_size => 9,  # 6 # 3 # 9
#                 :hidden_layers => 2,
#                 :hidden_neurons => 64,  # 64
#                 :dropout => 0.0,
#                 :activation_function => :relu,  # :relu
#                 :last_activation_function => :softmax,  # (x) -> vcat(Flux.softmax(@view x[1:3, :]), Flux.softmax(@view x[4:6, :])) # [(:softmax, 3), (:softmax, 3)] # [(:softmax, 3), (:tanh, 1)],
#                 :loss => :kldivergence
#             )
#         ),
#     ),
# )
