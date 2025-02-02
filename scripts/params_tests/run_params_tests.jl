# IMPORTANT !!!
# I should make clean start every time - I should run script always from a new repl
if isdefined(Main, :CAN_RUN_ONLY_ONCE)
    throw("This script can be run only once, please restart julia session (REPL)")
else
    CAN_RUN_ONLY_ONCE = true
end

# I shoudn't put this code in a module, it is easier to have it in main, then both remote and local code have the same path to functions etc.
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

# we will change these values globally for all tests
CONSTANTS_DICT[:run_config] = Dict(
    :max_generations => 10_000_000,  # 200
    :max_evaluations => 20_000,
    :log => false,
    :visualize_each_n_epochs => 0,
)

# Number of run tests per each combination of tested values
CASES_PER_TEST = 1

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
                    # :random_matrix_mode => [:rand, :randn],
                    # :factor => [0.1, 0.3, 0.5, 1.0],
                ),
            ),
        ),
    ),
    (
        :ContinuousStatesGroupingSimpleGA,
        Dict(
            :ContinuousStatesGroupingSimpleGA => Dict(
                :env_wrapper => Dict(
                    :n_clusters => [20, 40, 100],
                    :m_value => [1, 2]
                ),
                :fihc => Dict(
                    :fihc_mode => [:per_gene_rand],
                    :norm_mode => [:d_sum, :min_0],
                    :random_matrix_mode => [:rand, :randn],
                    :factor => [0.1, 0.3, 0.5, 1.0],
                    :genes_combination => [:hier, :flat],
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
CASES_RESULTS_FILE = "_cases_results_$(timestamp).log"
SCRIPTS_DIR_TO_COPY = joinpath(pwd(), "scripts") # copy constants.jl to logs dir, so that I know what were the exact settings when I ran it
SRC_DIR_TO_COPY = joinpath(pwd(), "src")  # copy src dir to logs dir, so that I know what was the code when I ran it

LOGS_DIR_RESULTS = joinpath(LOGS_DIR, "results")
LOGS_DIR_SRC = joinpath(LOGS_DIR, "src")
LOGS_DIR_ANALYSIS = joinpath(LOGS_DIR, "analysis")

# --------------------------------------------------------------------------------------------------
# Run the tests

DistributedEnvironments.@initcluster(CLUSTER_CONFIG_MAIN, CLUSTER_CONFIG_HOSTS, TMP_DIR_NAME, COPY_ENV_AND_CODE)

RELATIVE_PATH_TO_THIS_DIR = splitpath(@__DIR__)[length(splitpath(dirname(Base.active_project())))+1:end]

Distributed.@everywhere begin
    # important things to improve performance on intel CPUs:
    using MKL
    using LinearAlgebra
    import Distributed
    import Pkg
    import Logging
    import Dates
    import Random
    import ProgressMeter
    import DataFrames
    import CSV

    seed = time_ns() ⊻ UInt64(hash(Distributed.myid())) # xor between time nano seconds and hash of worker id
    Random.seed!(seed)

    if Distributed.myid() == 1  # I could run one version for all, but I want to make linter happy with normal paths :)
        include("../../src/JuliaEvolutionaryCars.jl")
        import .JuliaEvolutionaryCars
        include("../custom_loggers.jl")
        import .CustomLoggers
        include("_tests_utils.jl")
    else
        current_absolute_dir = joinpath(dirname(Base.active_project()), ($RELATIVE_PATH_TO_THIS_DIR)...)
        blas_threads = parse(Int, ENV["JULIA_BLAS_THREADS"])
        BLAS.set_num_threads(blas_threads)
        include(joinpath(current_absolute_dir, "../../src/JuliaEvolutionaryCars.jl"))
        import .JuliaEvolutionaryCars
        include(joinpath(current_absolute_dir,"../custom_loggers.jl"))
        import .CustomLoggers
        include(joinpath(current_absolute_dir, "_tests_utils.jl"))
        # number of julia threads for main one doesnt make any difference, since it is not used
        text = (
            "Worker $(Distributed.myid()) started at $(Dates.now()) with seed: $seed\n" *
            "Project name: $(Pkg.project().name)\n" *
            "BLAS kernel: $(BLAS.get_config())\n" *
            "Number of BLAS threads: $(BLAS.get_num_threads())\n" *
            "Number of Julia threads: $(Threads.nthreads())\n"
        )
        println(text)
    end
end

mkpath(LOGS_DIR)
mkpath(LOGS_DIR_RESULTS)
mkpath(LOGS_DIR_SRC)
mkpath(LOGS_DIR_ANALYSIS)
println("Logs will be saved in: $LOGS_DIR")
cp(SCRIPTS_DIR_TO_COPY, joinpath(LOGS_DIR_SRC, "scripts"), force=true)
println("Copied scripts dir to logs dir")
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

special_dicts_with_cases = create_all_special_dicts_with_cases(special_dicts, CASES_PER_TEST)
special_dicts_with_cases = consider_done_cases(special_dicts_with_cases, LOGS_DIR_RESULTS)

Logging.@info "Starting computation"
remote_channel = Distributed.RemoteChannel(() -> Channel{RemoteResult}(RESULT_CHANNEL_BUFFER_SIZE))
result_info = [FinalResultLog(save_name(entry...), false, "Not yet computed") for entry in special_dicts_with_cases]

cases_results_path = joinpath(LOGS_DIR, CASES_RESULTS_FILE)

channel_controller_task = @async run_channel_controller!(remote_channel, result_info, LOGS_DIR_RESULTS, cases_results_path)

rand_perm_special_dicts_with_cases = Random.shuffle(collect(enumerate(special_dicts_with_cases)))  # they will process in random order

Distributed.@everywhere CONSTANTS_DICT_LOCAL_ON_WORKER = deepcopy($CONSTANTS_DICT)
# results_trash itself is not used, hence the name
results_trash = ProgressMeter.@showprogress Distributed.pmap(rand_perm_special_dicts_with_cases; retry_delays = zeros(3)) do entry
    task_id, one_special_dict_with_case = entry
    remote_run(one_special_dict_with_case, CONSTANTS_DICT_LOCAL_ON_WORKER, task_id, remote_channel)  # this constants dict is deepcopied inside that function
end

put!(remote_channel, RemoteResult("", "", :stop, "Finished", 1, -1))
wait(channel_controller_task)

Logging.@info construct_text_from_final_results(result_info)
DistributedEnvironments.remove_workers!()
