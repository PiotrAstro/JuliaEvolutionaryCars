using MKL
using LinearAlgebra
BLAS.set_num_threads(1)
println("BLAS kernel: $(BLAS.get_config())")
println("Number of BLAS threads: $(BLAS.get_num_threads())")
println("Number of Julia threads: $(Threads.nthreads())")


include("../custom_loggers.jl")
import .CustomLoggers
import Logging
Logging.global_logger(CustomLoggers.PlainInfoLogger())

include("../constants.jl")
include("../../src/JuliaEvolutionaryCars.jl")
import .JuliaEvolutionaryCars

CONSTANTS_DICT[:run_config][:max_generations] = 1
CONSTANTS_DICT[:Evolutionary_Mutate_Population][:population_size] = 5
CONSTANTS_DICT[:ContinuousStatesGroupingDE][:individuals_n] = 5
JuliaEvolutionaryCars.run(:Evolutionary_Mutate_Population, CONSTANTS_DICT)
JuliaEvolutionaryCars.run(:ContinuousStatesGroupingDE, CONSTANTS_DICT)
