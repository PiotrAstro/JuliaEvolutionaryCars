# important things to improve performance on intel CPUs:
using MKL
using LinearAlgebra
# BLAS.set_num_threads(4)
println("BLAS kernel: $(BLAS.get_config())")
println("Number of BLAS threads: $(BLAS.get_num_threads())")
println("Number of Julia threads: $(Threads.nthreads())")


include("custom_loggers.jl")
import .CustomLoggers
import Logging
Logging.global_logger(CustomLoggers.PlainInfoLogger())


include("constants.jl")
include("../src/JuliaEvolutionaryCars.jl")
import .JuliaEvolutionaryCars

# Run the algorithm

# JuliaEvolutionaryCars.run_EvMutPop(CONSTANTS_DICT)
JuliaEvolutionaryCars.run(:StatesGroupingGA, CONSTANTS_DICT)
# JuliaEvolutionaryCars.run_ConStGroup(CONSTANTS_DICT)