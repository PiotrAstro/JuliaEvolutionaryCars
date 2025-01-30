module ParamsTests

import Distributed
import Dates
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

include("../constants.jl")
include("cluster/cluster_config.jl")
include("cluster/DistributedEnvironments.jl")
import .DistributedEnvironments

"""
How to set TESTED_VALUES:
    It should be:
    TESTED_VALUES = Vector{
        Tuple{
            Symbol,  # optimizer
            Dict{Symbol, Dict{Symbol, Any}}  # special_dict
        }
    }

    final values should be in a list
    TESTED_VALUES = [
        (
            :StatesGroupingGA,
            Dict(
                :StatesGroupingGA => Dict(
                    :nn_autoencoder => Dict(
                        :mmd_weight => [0.0, 0.1, 1.0],
                    ),
                    :fuzzy_logic_of_n_closest => [1, 5],
                )
            ),
        ),
    ]


    If I want to change one value throught all tests or use list of values:
    TESTED_VALUES = [
        (
            :StatesGroupingGA,
            Dict(
                :StatesGroupingGA => Dict(
                    :list_parameter [[1, 2, 3], [4, 5, 6]],
                    :set_one_to_all => [1],
                )
            ),
        ),
    ]

    I can test multiple optimizers or multiple parameters for one optimizer:
    TESTED_VALUES = [
        (
            :StatesGroupingGA,
            Dict(
                ...  # parameters for StatesGroupingGA
            ),
        ),
        (
            :StatesGroupingGA,
            Dict(
                ...  # separate test for StatesGroupingGA
            ),
        ),
        (
            :EvolutionaryMutatePopulation,
            Dict(
                ...  # parameters for EvolutionaryMutatePopulation
            ),
        ),
    ]
"""

# --------------------------------------------------------------------------------------------------
# Start of real settings
# --------------------------------------------------------------------------------------------------

RUN_ALL_CASES_ON_ONE_WORKER = true  # if true, all cases will be run on one worker, if false, cases will be distributed among workers
# if my jobs are small than it should be true, if they are big, it should be false

# we will change these values globally for all tests
CONSTANTS_DICT[:run_config] = Dict(
    :max_generations => 10_000_000,  # 200
    :max_evaluations => 20_000,
    :log => false,
    :visualize_each_n_epochs => 0,
)

# Number of run tests per each combination of tested values
CASES_PER_TEST = 50

LOGS_DIR = joinpath(pwd(), "log", "parameters_tests_" * timestamp)  # running test from scratch
# LOGS_DIR = joinpath(pwd(), "log", "parameters_tests_2024-12-27_12-31-13")  # running test from some start_position - it will recognize already done cases

# Values that will be tested
TESTED_VALUES = [
    (
        :ContinuousStatesGroupingSimpleGA,
        Dict(
            :ContinuousStatesGroupingSimpleGA => Dict(
                :env_wrapper => Dict(
                    :n_clusters => [20, 40, 100],
                    :m_value => [1, 2]
                ),
                :fihc => Dict(
                    :fihc_mode => [:matrix_rand],
                    :norm_mode => [:d_sum, :min_0],
                    :random_matrix_mode => [:rand, :randn],
                    :factor => [0.1, 0.3, 0.5, 1.0],
                ),
            ),
        ),
    ),
    # (
    #     :ContinuousStatesGroupingSimpleGA,
    #     Dict(
    #         :ContinuousStatesGroupingSimpleGA => Dict(
    #             :env_wrapper => Dict(
    #                 :n_clusters => [20, 40, 100],
    #                 :m_value => [1, 2]
    #             ),
    #             :fihc => Dict(
    #                 :fihc_mode => [:per_gene_rand],
    #                 :norm_mode => [:d_sum, :min_0],
    #                 :random_matrix_mode => [:rand, :randn],
    #                 :factor => [0.1, 0.3, 0.5, 1.0],
    #                 :genes_combination => [:hier, :flat],
    #             ),
    #         ),
    #     ),
    # ),
    # (
    #     :ContinuousStatesGroupingSimpleGA,
    #     Dict(
    #         :ContinuousStatesGroupingSimpleGA => Dict(
    #             :env_wrapper => Dict(
    #                 :n_clusters => [20, 40, 100],
    #                 :m_value => [1, 2]
    #             ),
    #             :fihc => Dict(
    #                 :fihc_mode => [:disc_fihc],
    #                 :genes_combination => [:hier, :flat],
    #             ),
    #         ),
    #     ),
    # ),
    # (
    #     :ContinuousStatesGroupingSimpleGA,
    #     Dict(
    #         :ContinuousStatesGroupingSimpleGA => Dict(
    #             :env_wrapper => Dict(
    #                 :n_clusters => [20, 40, 100],
    #                 :m_value => [1, 2]
    #             ),
    #             :fihc => Dict(
    #                 :fihc_mode => [:fihc_cont],
    #                 :norm_mode => [:d_sum, :min_0],
    #                 :factor => [0.1, 0.3, 0.5, 1.0],
    #                 :genes_combination => [:hier, :flat],
    #             ),
    #         ),
    #     ),
    # ),
]

# --------------------------------------------------------------------------------------------------
# End of real settings
# --------------------------------------------------------------------------------------------------

# Rather constant settings:
OUTPUT_LOG_FILE = "_output_$(timestamp).log"
SCRIPTS_DIR_TO_COPY = joinpath(pwd(), "scripts") # copy constants.jl to logs dir, so that I know what were the exact settings when I ran it
SRC_DIR_TO_COPY = joinpath(pwd(), "src")  # copy src dir to logs dir, so that I know what was the code when I ran it

LOGS_DIR_RESULTS = joinpath(LOGS_DIR, "results")
LOGS_DIR_SRC = joinpath(LOGS_DIR, "src")
LOGS_DIR_ANALYSIS = joinpath(LOGS_DIR, "analysis")

# --------------------------------------------------------------------------------------------------
# Run the tests
"""
Set proper number of separate workers from main worker.
"""
function set_proper_workers(cluster_config_main, cluster_config_hosts)
    if length(Distributed.procs()) > 1
        workers_to_remove_ids = [pid for pid in Distributed.workers() if pid != 1]
        Distributed.rmprocs(workers_to_remove_ids...)
        println("Removed workers $workers_to_remove_ids")
    end
    
    # adding main process
    println("Adding main host -> $(cluster_config_main[:use_n_workers]) workers")
    env = Dict(
        "JULIA_NUM_THREADS" => "$(cluster_config_main[:julia_threads_per_worker])",
        "JULIA_BLAS_THREADS" => "$(cluster_config_main[:blas_threads_per_worker])"
    )
    enable_threaded_blas = cluster_config_main[:blas_threads_per_worker] > 1
    Distributed.addprocs(cluster_config_main[:use_n_workers], env=env, enable_threaded_blas=enable_threaded_blas, topology=:master_worker)
    println("Added main host\n")
    
    for host in cluster_config_hosts
        address = "$(host[:username])@$(host[:host_ip])"
        println("Adding $address host -> $(host[:use_n_workers]) workers")
        env = Dict(
            "JULIA_NUM_THREADS" => "$(host[:julia_threads_per_worker])",
            "JULIA_BLAS_THREADS" => "$(host[:blas_threads_per_worker])"
        )
        sshflags = `-i $(host[:private_key_path]) -p $(host[:port])`
        enable_threaded_blas = host[:blas_threads_per_worker] > 1
        Distributed.addprocs(
            [(address, host[:use_n_workers])];
            sshflags=sshflags,
            env=env,
            shell=host[:shell],
            tunnel=false,
            enable_threaded_blas=enable_threaded_blas,
            topology=:master_worker,
            dir=host[:dir],
            exename=host[:exe_path],
        )
        println("Added $address host\n")
    end
    
    workers_pids = Distributed.workers()
    println("\n\nAdded workers: $workers_pids")
end

DistributedEnvironments.initcluster(CLUSTER_CONFIG_MAIN, CLUSTER_CONFIG_HOSTS)

Distributed.@everywhere begin
    # important things to improve performance on intel CPUs:
    import Pkg
    println(Pkg.project().name)
    using MKL
    using LinearAlgebra
    import Distributed
    # take blas threads from env variable
    blas_threads = get(ENV, "JULIA_BLAS_THREADS", 1)
    BLAS.set_num_threads(blas_threads)

    import CSV
    import DataFrames
    import Logging
    import Dates
    import Random
    import ProgressMeter
    seed = time_ns() ⊻ UInt64(hash(Distributed.myid())) # xor between time nano seconds and hash of worker id
    Random.seed!(seed)
    include("../../src/JuliaEvolutionaryCars.jl")
    import .JuliaEvolutionaryCars

    include("../custom_loggers.jl")
    import .CustomLoggers

    include("_tests_utils.jl")

    # number of julia threads for main one doesnt make any difference, since it is not used
    text = (
        "Worker $(Distributed.myid()) started at $(Dates.now()) with seed: $seed\n" *
        "BLAS kernel: $(BLAS.get_config())\n" *
        "Number of BLAS threads: $(BLAS.get_num_threads())\n" *
        "Number of Julia threads: $(Threads.nthreads())\n"
    )
    println(text)
end

function run_params_tests()
    mkpath(LOGS_DIR)
    mkpath(LOGS_DIR_RESULTS)
    mkpath(LOGS_DIR_SRC)
    mkpath(LOGS_DIR_ANALYSIS)
    println("Logs will be saved in: $LOGS_DIR")
    cp(SCRIPTS_DIR_TO_COPY, joinpath(LOGS_DIR_SRC, "scripts"), force=true)
    println("Copied scripts to logs dir")
    cp(SRC_DIR_TO_COPY, joinpath(LOGS_DIR_SRC, "src"), force=true)
    println("Copied src dir to logs dir")

    file_logger = CustomLoggers.SimpleFileLogger(joinpath(LOGS_DIR, OUTPUT_LOG_FILE), true)
    Logging.global_logger(file_logger)

    special_dicts = create_all_special_dicts(TESTED_VALUES)

    io = IOBuffer()
    show(io, MIME"text/plain"(), TESTED_VALUES)
    log_text = "Tested values settings:\n" * String(take!(io))
    show(io, MIME"text/plain"(), special_dicts)
    log_text *= "\n\nSpecial dicts settings:\n" * String(take!(io))
    Logging.@info log_text

    special_dicts_no_cases = [
        (optimizer, special_dict, deepcopy(CONSTANTS_DICT))
        for (optimizer, special_dict) in vec([(optimizer, special_dict) for (optimizer, special_dict) in special_dicts])
    ]
    if RUN_ALL_CASES_ON_ONE_WORKER
        special_dicts_with_cases = [
            (optimizer, special_dict, config_copy, [i for i in 1:CASES_PER_TEST])
            for (optimizer, special_dict, config_copy) in special_dicts_no_cases
        ]
    else
        special_dicts_with_cases = vec([
            (optimizer, special_dict, config_copy, [case])
            for (optimizer, special_dict, config_copy) in special_dicts_no_cases, case in 1:CASES_PER_TEST
        ])
    end
    consider_done_cases!(special_dicts_with_cases, LOGS_DIR_RESULTS)

    Logging.@info "Starting computation"
    results_lists = ProgressMeter.@showprogress Distributed.pmap(eachindex(special_dicts_with_cases)) do i 
        Logging.global_logger(CustomLoggers.RemoteLogger())
        optimizer, special_dict, config_copy, cases = special_dicts_with_cases[i]
        result_list = Vector{Bool}(undef, length(cases))
        for (j, case) in enumerate(cases)
            try
                config_copy_copy = deepcopy(config_copy)
                special_dict_copy = deepcopy(special_dict)
                result_list[j] = run_one_test(optimizer, special_dict_copy, config_copy_copy, case, LOGS_DIR_RESULTS, Distributed.myid())
            catch e
                Logging.@error "workerid $(Distributed.myid()) failed with error: $e"
                result_list[j] = false
            end
        end
        return result_list
    end

    texts = []
    for (results_list, (optimizer, special_dict, _, cases)) in zip(results_lists, special_dicts_with_cases)
        for (result, case) in zip(results_list, cases)
            push!(
                texts,
                (result ? "success" : "failed") * "  ->  " * save_name(optimizer, special_dict, case)
            )
        end
    end

    text_log = "Finished computation, result:\n" * join(texts, "\n")
    Logging.@info text_log
end

end  # module ParamsTests

import .ParamsTests
ParamsTests.run_params_tests()