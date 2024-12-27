import Distributed
import Dates

include("../constants.jl")

# --------------------------------------------------------------------------------------------------
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
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

USE_N_WORKERS = 8  # how many workers to use, main worker is worker 1 and is not included - it doesnt perform calculations
BLAS_THREADS_PER_WORKER = 8
JULIA_THREADS_PER_WORKER = 4

CASES_PER_TEST = 1
OUTPUT_LOG_FILE = "_output_$(timestamp)_.log"

# running test from scratch
LOGS_DIR = joinpath(pwd(), "log", "parameters_tests_" * Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS"))
# running test from some start_position
# LOGS_DIR = joinpath(pwd(), "log", "parameters_tests_2024-12-27_12-31-13")

# we will change these values globally for all tests
CONSTANTS_DICT[:run_config] = Dict(
    :max_generations => 200,  # 200
    :max_evaluations => 1_000_000,
    :log => false,
    :visualize_each_n_epochs => 0,
)

# Values that will be tested
TESTED_VALUES = [
    (
        :StatesGroupingGA,
        Dict(
            :StatesGroupingGA => Dict(
                :env_wrapper => Dict(
                    :n_clusters => [40, 100],  # 40 or 100
                    :distance_metric => [:cosine, :euclidean],  # :euclidean or :cosine or :cityblock
                    :exemplars_clustering => [:genie, :kmedoids, :pam],  # :genie or :pam or :kmedoids
                )
            )
        ),
    ),
]


# --------------------------------------------------------------------------------------------------
# Run the tests

"""
Set proper number of separate workers from main worker.
"""
function set_proper_workers(workers_n, julia_threads)
    if length(Distributed.procs()) > 1
        for pid in Distributed.workers()
            Distributed.rmprocs(pid)
            println("Removed worker $pid")
        end
    end
    
    Distributed.addprocs(workers_n, exeflags=["--threads=$julia_threads"])
    workers_pids = Distributed.workers()
    println("Added workers: $workers_pids")
end
set_proper_workers(USE_N_WORKERS, JULIA_THREADS_PER_WORKER)

Distributed.@everywhere begin
    # important things to improve performance on intel CPUs:
    using MKL
    using LinearAlgebra
    BLAS.set_num_threads($BLAS_THREADS_PER_WORKER)

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

mkpath(LOGS_DIR)
println("Logs will be saved in: $LOGS_DIR")

file_logger = CustomLoggers.SimpleFileLogger(joinpath(LOGS_DIR, OUTPUT_LOG_FILE), true)
Logging.global_logger(file_logger)

special_dicts = create_all_special_dicts(TESTED_VALUES)

io = IOBuffer()
show(io, MIME"text/plain"(), TESTED_VALUES)
log_text = "Tested values settings:\n" * String(take!(io))
show(io, MIME"text/plain"(), special_dicts)
log_text *= "\n\nSpecial dicts settings:\n" * String(take!(io))
Logging.@info log_text

special_dicts_with_cases = [
    (optimizer, special_dict, deepcopy(CONSTANTS_DICT), i)
    for (optimizer, special_dict, i) in vec([(optimizer, special_dict, i) for (optimizer, special_dict) in special_dicts, i in 1:CASES_PER_TEST])
]
consider_done_cases!(special_dicts_with_cases, LOGS_DIR)

results = ProgressMeter.@showprogress Distributed.pmap(eachindex(special_dicts_with_cases)) do i 
    Logging.global_logger(CustomLoggers.RemoteLogger())
    try
        optimizer, special_dict, config_copy, case = special_dicts_with_cases[i]
        run_one_test(optimizer, special_dict, config_copy, case, LOGS_DIR, Distributed.myid())
        return true
    catch e
        Logging.@error "workerid $(Distributed.myid()) failed with error: $e"
        return false
    end
end



texts = [
    (result ? "success" : "failed") * "  ->  " * save_name(optimizer, special_dict, case)
    for (result, (optimizer, special_dict, _, case)) in zip(results, special_dicts_with_cases)
]

text_log = "Finished computation, result:\n" * join(texts, "\n")
Logging.@info text_log