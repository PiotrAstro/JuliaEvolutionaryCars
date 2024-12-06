# important things to improve performance on intel CPUs:
using MKL
using LinearAlgebra
println("BLAS kernel: $(BLAS.get_config())")
println("Number of BLAS threads: $(BLAS.get_num_threads())")
println("Number of Julia threads: $(Threads.nthreads())")


include("constants.jl")
include("src/JuliaEvolutionaryCars.jl")
import .JuliaEvolutionaryCars

# Run the algorithm
# Test config dict - it will alter values in the constants.jl file to make some tests



# JuliaEvolutionaryCars.run_EvMutPop(CONSTANTS_DICT)
JuliaEvolutionaryCars.run_StGroupGA(CONSTANTS_DICT)
# JuliaEvolutionaryCars.run_ConStGroup(CONSTANTS_DICT)