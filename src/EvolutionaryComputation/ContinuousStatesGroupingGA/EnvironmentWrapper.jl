module EnvironmentWrapper

import ..NeuralNetwork
import ..Environment
import ..StatesGrouping

import Statistics
import Dates
import Logging
import Printf

export EnvironmentWrapperStruct, get_action_size, get_groups_number, get_fitness, copy, is_verbose, set_verbose!, translate, create_new_based_on, create_time_distance_tree, normalize_genes_min_0!, clean!

# --------------------------------------------------------------------------------------------------
# Structs

mutable struct StructMemory
    _distance_membership_levels::Vector{Vector{Vector{Float32}}}
    _raw_exemplars::NeuralNetwork.AbstractStateSequence
    _decoder::NeuralNetwork.AbstractNeuralNetwork
    _autoencoder::NeuralNetwork.AbstractNeuralNetwork
end

mutable struct EnvironmentWrapperStruct
    _envs::Vector{<:Environment.AbstractEnvironment}
    _n_clusters::Int
    _encoder::NeuralNetwork.AbstractNeuralNetwork
    _encoded_exemplars::Matrix{Float32}
    _distance_membership_levels_method::Symbol
    _max_states_considered::Int
    _distance_metric::Symbol  # :euclidean or :cosine or :cityblock
    _exemplars_clustering::Symbol  # :genieclust or :pam or :kmedoids
    _hclust_distance::Symbol  # :ward or :single or :complete or :average
    _hclust_time::Symbol  # :ward or :single or :complete or :average
    _exemplar_nn_dict
    _create_time_distance_tree
    _verbose::Bool
    _initial_space_explorers_n::Int
    _struct_memory::Union{StructMemory, Nothing}
    _run_statistics::Environment.RunStatistics
end



# --------------------------------------------------------------------------------------------------
# Public functions

function EnvironmentWrapperStruct(
        envs::Vector{<:Environment.AbstractEnvironment},
        run_statistics::Environment.RunStatistics=Environment.RunStatistics(),

        ; encoder_dict::Dict{Symbol,Any},
        decoder_dict::Dict{Symbol,Any},
        autoencoder_dict::Dict{Symbol,<:Any},
        initial_space_explorers_n::Int,
        max_states_considered::Int,
        n_clusters::Int,
        distance_metric::Symbol=:cosine,
        exemplars_clustering::Symbol=:genieclust,
        distance_membership_levels_method::Symbol=:hclust_complete,
        hclust_distance::Symbol=:ward,
        hclust_time::Symbol=:ward,
        time_distance_tree::Symbol=:mine,  # :mine or :markov
        exemplar_nn=Dict(
            :interaction_method=>:cosine,
            :membership_normalization=>:softmax,
            :activation_function=>:softmax,
        ),
        verbose::Bool=false
    )
    if time_distance_tree == :mine
        _create_time_distance_tree = StatesGrouping.create_time_distance_tree_mine
    elseif time_distance_tree == :markov
        _create_time_distance_tree = StatesGrouping.create_time_distance_tree_markov_fundamental
    else
        throw(ArgumentError("time_distance_tree must be :mine or :markov"))
    end

    encoder = NeuralNetwork.get_neural_network(encoder_dict[:name])(; encoder_dict[:kwargs]...)
    decoder = NeuralNetwork.get_neural_network(decoder_dict[:name])(; decoder_dict[:kwargs]...)
    autoencoder = NeuralNetwork.Autoencoder(encoder, decoder; autoencoder_dict...)

    # initial state space exploration
    # random NNs creation
    action_n = Environment.get_action_size(envs[1])
    NNs = [
        NeuralNetwork.Random_NN(action_n) for _ in 1:initial_space_explorers_n
    ]
    # states collection
    trajectories = _collect_trajectories(envs, NNs, run_statistics)
    states = _combine_states_from_trajectories([(1.0, trajectories)], max_states_considered)
    NeuralNetwork.learn!(autoencoder, states; verbose=verbose)

    if verbose
        Logging.@info "Autoencoder trained"
    end

    encoded_states = NeuralNetwork.predict(encoder, states)
    exemplars_ids, _ = StatesGrouping.get_exemplars(
        encoded_states,
        n_clusters;
        distance_metric=distance_metric,
        exemplars_clustering=exemplars_clustering,
        hclust_distance=hclust_distance
    )
    encoded_exemplars = encoded_states[:, exemplars_ids]
    distance_membership_levels = StatesGrouping.distance_membership_levels(
        encoded_states,
        encoded_exemplars; 
        distance_metric=distance_metric,
        method=distance_membership_levels_method
    )
    states_exeplars = NeuralNetwork.get_sequence_with_ids(states, exemplars_ids)

    # ---------------------------------------------
    # tmp stuff
    # for method in [
    #     :flat, :hclust_complete, :hclust_single,
    #     :kmeans_exemplars_crisp, :kmeans_all_crisp, :pam_exemplars_crisp, :pam_all_crisp,
    #     :kmeans_exemplars_fuzzy, :kmeans_all_fuzzy, :pam_exemplars_fuzzy, :pam_all_fuzzy
    # ]
    #     result = StatesGrouping.distance_membership_levels(
    #         encoded_states,
    #         encoded_exemplars; 
    #         distance_metric=distance_metric,
    #         method=method,
    #         mval=m_value
    #     )
    #     text = ""
    #     for res_lev in result
    #         text *= "\n[\n"
    #         for node in res_lev
    #             text *= "\t[" * join([Printf.@sprintf("%.2f", memb) for memb in node], "  ") * "]\n"
    #         end
    #         text *= "]\n"
    #     end
    #     # save it to file
    #     open("z__$method.txt", "w") do f
    #         write(f, text)
    #     end
    # end
    # throw("dfrvdv")
    # ---------------------------------------------

    if verbose
        Logging.@info "Exemplars tree created"
    end

    struct_memory = StructMemory(
        distance_membership_levels,
        states_exeplars,
        decoder,
        autoencoder,
    )
    return (
        EnvironmentWrapperStruct(
            envs,
            n_clusters,
            encoder,
            encoded_exemplars,
            distance_membership_levels_method,
            max_states_considered,
            distance_metric,
            exemplars_clustering,
            hclust_distance,
            hclust_time,
            exemplar_nn,
            _create_time_distance_tree,
            verbose,
            initial_space_explorers_n,
            struct_memory,
            run_statistics
        ),
        states
    )
end


"""
Remove things that are not needed for normal inference, reduces memory
"""
function clean!(env_wrap::EnvironmentWrapperStruct)
    env_wrap._struct_memory = nothing
end

function random_reinitialize_exemplars!(env_wrap::EnvironmentWrapperStruct, n_clusters::Int=env_wrap._n_clusters)
    action_n = Environment.get_action_size(env_wrap._envs[1])
    NNs = [
        NeuralNetwork.Random_NN(action_n) for _ in 1:env_wrap._initial_space_explorers_n
    ]
    # states collection
    trajectories = _collect_trajectories(env_wrap._envs, NNs, env_wrap._run_statistics)
    states = _combine_states_from_trajectories([(1.0, trajectories)], env_wrap._max_states_considered)

    env_wrap._n_clusters = n_clusters
    states_nn_input = states
    encoded_states = NeuralNetwork.predict(env_wrap._encoder, states_nn_input)
    exemplars_ids, _ = StatesGrouping.get_exemplars(
        encoded_states,
        n_clusters;
        distance_metric=env_wrap._distance_metric,
        exemplars_clustering=env_wrap._exemplars_clustering,
        hclust_distance=env_wrap._hclust_distance
    )
    env_wrap._encoded_exemplars = encoded_states[:, exemplars_ids]
    env_wrap._struct_memory._distance_membership_levels = StatesGrouping.distance_membership_levels(
        encoded_states,
        env_wrap._encoded_exemplars; 
        distance_metric=env_wrap._distance_metric,
        method=env_wrap._distance_membership_levels_method
    )
    env_wrap._struct_memory._raw_exemplars = NeuralNetwork.get_sequence_with_ids(states, exemplars_ids)
end

"""
returns Tuple{Vector{Trajectory}, TreeNode}
"""
function create_time_distance_tree(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32})
    trajectories = _collect_trajectories(env_wrap._envs, [get_full_NN(env_wrap, translation)], env_wrap._run_statistics)
    states_in_trajectories = [trajectory.states for trajectory in trajectories]
    full_nn = get_full_NN(env_wrap, translation)
    memberships_by_trajectory = [NeuralNetwork.membership(full_nn, states_one_traj) for states_one_traj in states_in_trajectories]
    return trajectories, env_wrap._create_time_distance_tree(memberships_by_trajectory, env_wrap._hclust_time)
end

function get_trajectories(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32})
    return _collect_trajectories(env_wrap._envs, [get_full_NN(env_wrap, translation)], env_wrap._run_statistics)
end

function copy(env_wrap::EnvironmentWrapperStruct)::EnvironmentWrapperStruct
    envs_copy = [Environment.copy(env) for env in env_wrap._envs]
    autoencoder_copy = NeuralNetwork.copy(env_wrap._struct_memory._autoencoder)
    encoder_copy = autoencoder_copy.encoder
    decoder_copy = autoencoder_copy.decoder
    struct_memory_copy = StructMemory(
        env_wrap._struct_memory._distance_membership_levels,
        env_wrap._struct_memory._raw_exemplars,
        decoder_copy,
        autoencoder_copy,
    )

    return EnvironmentWrapperStruct(
        envs_copy,
        env_wrap._n_clusters,
        encoder_copy,
        env_wrap._encoded_exemplars,
        env_wrap._distance_membership_levels_method,
        env_wrap._max_states_considered,
        env_wrap._distance_metric,
        env_wrap._exemplars_clustering,
        env_wrap._hclust_distance,
        env_wrap._hclust_time,
        env_wrap._exemplar_nn_dict,
        env_wrap._create_time_distance_tree,
        env_wrap._verbose,
        env_wrap._initial_space_explorers_n,
        struct_memory_copy,
        env_wrap._run_statistics
    )
end


"""
It normalizes genes by making it non negative and sum to 0
opposed to normalize_genes_min_0! if input is e.g. 0.2 0.4 0.4 it will stay the same
"""
function normalize_genes!(genes::Matrix{Float32})
    for col in eachcol(genes)
        min_value = minimum(col)
        if min_value < 0
            col .+= abs(min_value)
        end
        col ./= sum(col)
    end
end

"""
It normalizes genes by subtracting smallest value from each col and then subtracting by sum
if input is e.g. 0.2 0.4 0.4 it will be normalized to 0 0.5 0.5
"""
function normalize_genes_min_0!(genes::Matrix{Float32})
    for col in eachcol(genes)
        normalize_genes_min_0!(col)
    end
end

function normalize_genes_min_0!(gene::AbstractVector{Float32})
    min_value = minimum(gene)
    gene .-= min_value
    gene ./= sum(gene)
end

function get_action_size(env_wrap::EnvironmentWrapperStruct)::Int
    return Environment.get_action_size(env_wrap._envs[1])
end

function get_groups_number(env_wrap::EnvironmentWrapperStruct)::Int
    return size(env_wrap._encoded_exemplars, 2)
end

function get_fitness(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32}) :: Float64
    full_NN = get_full_NN(env_wrap, translation)

    envs_copies = [Environment.copy(env) for env in env_wrap._envs]
    result = sum(Environment.get_trajectory_rewards!(envs_copies, full_NN; run_statistics=env_wrap._run_statistics, reset=true))

    return result
end


"""
Create new env_wrapper based on the trajectories and percentages of them.
It will take percent of internal max states considered from each group of trajectories,

args:
env_wrap::EnvironmentWrapperStruct,
trajectories_and_percentages::Vector{Tuple{Float64, Vector{<:Environment.Trajectory}}}

kwargs:
new_n_clusters::Int=-1,

this vector sometimes has a problem with casting on callee side, so we will cast it here
States in trajectories are equal - each states has the same chance of beeing picked (the are added to the same array).

In the future, one can adapt some values of env_wrapper e.g. n_clusters, max_states_considered, fuzzy_logic_of_n_closest etc.
"""
function create_new_based_on(
    env_wrap::EnvironmentWrapperStruct,
    trajectories_and_percentages::Vector{<:Any};
    new_n_clusters::Int=-1,
)
    if new_n_clusters == -1
        new_n_clusters = env_wrap._n_clusters
    end
    TSEQ = typeof(trajectories_and_percentages[1][2][1].states) # type of states sequences
    trajectories_and_percentages_casted = Vector{Tuple{Float64,Vector{Environment.Trajectory{TSEQ}}}}(trajectories_and_percentages)

    new_env_wrapper = copy(env_wrap)

    new_states = _combine_states_from_trajectories(trajectories_and_percentages_casted, new_env_wrapper._max_states_considered)
    NeuralNetwork.learn!(new_env_wrapper._struct_memory._autoencoder, new_states; verbose=new_env_wrapper._verbose)

    if env_wrap._verbose
        Logging.@info "Autoencoder retrained"
    end

    new_encoded_states = NeuralNetwork.predict(new_env_wrapper._encoder, new_states)

    # # get new exemplars, states and newly encoded states
    new_exemplars_ids, _ = StatesGrouping.get_exemplars(new_encoded_states, new_n_clusters; distance_metric=env_wrap._distance_metric, exemplars_clustering=env_wrap._exemplars_clustering, hclust_distance=env_wrap._hclust_distance)
    new_exemplars = new_encoded_states[:, new_exemplars_ids]
    new_distance_membership_levels = StatesGrouping.distance_membership_levels(
        new_encoded_states,
        new_exemplars; 
        distance_metric=env_wrap._distance_metric,
        method=env_wrap._distance_membership_levels_method
    )
    new_raw_exemplars = NeuralNetwork.get_sequence_with_ids(new_states, new_exemplars_ids)

    new_env_wrapper._encoded_exemplars = new_exemplars
    new_env_wrapper._struct_memory._raw_exemplars = new_raw_exemplars
    new_env_wrapper._struct_memory._distance_membership_levels = new_distance_membership_levels
    new_env_wrapper._n_clusters = new_n_clusters

    return new_env_wrapper, new_states
end

function translate(
    from_env_wrap::EnvironmentWrapperStruct,
    from_translation::Matrix{Float32},
    to_env_wrap::EnvironmentWrapperStruct,
    to_genes_indices::Vector{Int}=collect(1:to_env_wrap._n_clusters)  # by default translates all genes
)::Matrix{Float32}  # translated genes, only to_genes_indices are present
    # create NN

    from_full_NN = get_full_NN(from_env_wrap, from_translation)
    to_raw_exemplars = NeuralNetwork.get_sequence_with_ids(to_env_wrap._struct_memory._raw_exemplars, to_genes_indices)
    return NeuralNetwork.predict(from_full_NN, to_raw_exemplars)
end

function get_full_NN(env_wrap::EnvironmentWrapperStruct, translation::Matrix{Float32})
    return NeuralNetwork.ExemplarBasedNN(
        env_wrap._encoder,
        env_wrap._encoded_exemplars,
        translation;
        env_wrap._exemplar_nn_dict...
    )
    # return NeuralNetwork.DistanceBasedClassificator(
    #     env_wrap._encoder,
    #     env_wrap._encoded_exemplars,
    #     translation,
    #     get_action_size(env_wrap),
    #     -1,
    #     env_wrap._distance_metric
    # )
end

function is_verbose(env_wrap::EnvironmentWrapperStruct)::Bool
    return env_wrap._verbose
end

function set_verbose!(env_wrap::EnvironmentWrapperStruct, verbose::Bool)
    env_wrap._verbose = verbose
end

# --------------------------------------------------------------------------------------------------
# Private functions

function _collect_trajectories(envs::Vector{E}, NNs::Vector{<:NeuralNetwork.AbstractNeuralNetwork}, run_statistics::Environment.RunStatistics)::Vector{Environment.Trajectory{SEQ}} where {SEQ<:NeuralNetwork.AbstractStateSequence,E<:Environment.AbstractEnvironment{SEQ}}
    trajectories = Vector{Vector{Environment.Trajectory{SEQ}}}(undef, length(NNs))

    Threads.@threads for i in 1:length(NNs)
        # for i in 1:length(NNs)
        envs_copy = [Environment.copy(env) for env in envs]
        nn = NNs[i]
        trajectories[i] = Environment.get_trajectory_data!(envs_copy, nn; run_statistics=run_statistics, reset=true)
    end
    trajectories_flat = reduce(vcat, trajectories)
    return trajectories_flat
end

function _combine_states_from_trajectories(trajectories_and_percentages::Vector{Tuple{Float64,Vector{Environment.Trajectory{SEQ}}}}, pick_states_n::Int)::SEQ where {SEQ<:NeuralNetwork.AbstractStateSequence}
    @assert sum([percentage for (percentage, _) in trajectories_and_percentages]) ≈ 1.0
    states_to_combine = Vector{SEQ}()

    for (percentage, trajectories) in trajectories_and_percentages
        states_to_pick_n = Int(round(percentage * pick_states_n))
        states_total_n = sum([NeuralNetwork.get_length(trajectory.states) for trajectory in trajectories])
        states_to_pick_n = min(states_to_pick_n, states_total_n)

        states_to_pick = rand(1:states_total_n, states_to_pick_n)
        sequences = Vector{SEQ}()

        from_number = 1
        for i in eachindex(trajectories)
            trajectory = trajectories[i]
            states_one_traj = trajectory.states
            to_number = from_number + NeuralNetwork.get_length(states_one_traj) - 1
            states_to_pick_local = collect(filter(x -> x >= from_number && x <= to_number, states_to_pick))
            states_to_pick_local .-= from_number - 1
            states_one_traj = NeuralNetwork.get_sequence_with_ids(states_one_traj, states_to_pick_local)
            push!(sequences, states_one_traj)
            from_number = to_number + 1
        end

        states_local_combined = SEQ(sequences)
        push!(states_to_combine, states_local_combined)
    end
    states_combined = SEQ(states_to_combine)

    return states_combined
end

end