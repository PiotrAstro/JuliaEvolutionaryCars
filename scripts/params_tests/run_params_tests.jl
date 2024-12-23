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
USE_N_WORKERS = 8  # how many workers to use, main worker is worker 1 and is not included - it doesnt perform calculations
BLAS_THREADS_PER_WORKER = 1

CASES_PER_TEST = 10
LOGS_DIR = joinpath(pwd(), "log", "parameters_tests_" * Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS"))
OUTPUT_LOG_FILE = "_output.log"

# we will change these values globally for all tests
CONSTANTS_DICT[:run_config] = Dict(
    :max_generations => 200,
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
                    :n_clusters => [40, 100],
                    :distance_metric => [:cosine, :euclidean],  # :euclidean or :cosine or :cityblock
                    :exemplars_clustering => [:genie, :kmedoids, :pam],  # :genie or :pam or :kmedoids
                )
            )
        ),
    ),
]


# --------------------------------------------------------------------------------------------------
# Run the tests

# check how many workers there are - rm if needed
# -1 worker case worker 1 doesnt do calculations, but is included in Distributed.workers()
function set_proper_workers(workers_n)
    real_processes_n = workers_n + 1
    difference_in_workers_n = length(Distributed.workers()) - real_processes_n
    if difference_in_workers_n < 0
        Distributed.addprocs(-difference_in_workers_n)
    elseif difference_in_workers_n > 0
        for pid in Distributed.workers()
            Distributed.rmprocs(pid)
            if length(Distributed.workers()) == real_processes_n
                break
            end
        end
    end
end
set_proper_workers(USE_N_WORKERS)

Distributed.@everywhere begin
    # important things to improve performance on intel CPUs:
    using MKL
    using LinearAlgebra

    import CSV
    import DataFrames
    import Logging
    import Dates
    import Random
    seed = time_ns() ⊻ UInt64(hash(Distributed.myid()))
    Random.seed!(seed)
    println("Worker $(Distributed.myid()) started at $(Dates.now()) with seed: $seed")

    include("../../src/JuliaEvolutionaryCars.jl")
    import .JuliaEvolutionaryCars

    include("../custom_loggers.jl")
    import .CustomLoggers

    include("_tests_utils.jl")
    BLAS.set_num_threads($BLAS_THREADS_PER_WORKER)
end

println("BLAS kernel: $(BLAS.get_config())")
println("Number of BLAS threads: $(BLAS.get_num_threads())")
println("Number of Julia threads: $(Threads.nthreads())")

mkpath(LOGS_DIR)
file_logger = CustomLoggers.SimpleFileLogger(joinpath(LOGS_DIR, OUTPUT_LOG_FILE))
Logging.global_logger(file_logger)

special_dicts = create_all_special_dicts(TESTED_VALUES)
io = IOBuffer()
display(io, special_dicts)
Logging.@info "\n\n will run with the following settings:\n" * String(take!(io))
special_dicts_with_cases = [(optimizer, special_dict, deepcopy(CONSTANTS_DICT), i) for (optimizer, special_dict) in special_dicts, i in 1:CASES_PER_TEST]

results = Distributed.pmap(eachindex(special_dicts_with_cases)) do i 
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

text_log = "\n\nFinished computation, result:\n" * join(texts, "\n")
Logging.@info text_log