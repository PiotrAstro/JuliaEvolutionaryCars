# important things to improve performance on intel CPUs:
using MKL
using LinearAlgebra
BLAS.set_num_threads(1)
println("BLAS kernel: $(BLAS.get_config())")
println("Number of BLAS threads: $(BLAS.get_num_threads())")
println("Number of Julia threads: $(Threads.nthreads())")

import CSV
import DataFrames
import Logging
import Dates

# --------------------------------------------------------------------------------------------------
include("../../src/JuliaEvolutionaryCars.jl")
import .JuliaEvolutionaryCars

include("../custom_loggers.jl")
import .CustomLoggers
Logging.global_logger(CustomLoggers.SimpleFileLogger("crash_log.log"))

include("../constants.jl")
include("_tests_utils.jl")

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

CASES_PER_TEST = 1
LOGS_DIR = "log/parameters_tests_" * Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS") * "/"

# we will change these values globally for all tests
CONSTANTS_DICT[:run_config] = Dict(
    :max_generations => 1,
    :max_evaluations => 1_000_000,
    :log => true,
    :visualize_each_n_epochs => 0,
)

# Values that will be tested
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



# --------------------------------------------------------------------------------------------------
# Run the tests
mkpath(LOGS_DIR)
special_dicts = create_all_special_dicts(TESTED_VALUES)
println("\n\n will run with the following settings:")
display(special_dicts)
special_dicts_with_cases = [(optimizer, special_dict, deepcopy(CONSTANTS_DICT), i) for (optimizer, special_dict) in special_dicts, i in 1:CASES_PER_TEST]

println("\n\n\n\n Running tests")


Threads.@threads for i in 1:length(special_dicts_with_cases)
# for i in 1:length(special_dicts_with_cases)
    optimizer, special_dict, config_copy, case = special_dicts_with_cases[i]
    run_one_test(optimizer, special_dict, config_copy, case, LOGS_DIR)
end
