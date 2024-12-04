include("constants.jl")
include("src/JuliaEvolutionaryCars.jl")

import .JuliaEvolutionaryCars

# Run the algorithm

# JuliaEvolutionaryCars.run_EvMutPop(CONSTANTS_DICT)
JuliaEvolutionaryCars.run_StGroupGA(CONSTANTS_DICT)
# JuliaEvolutionaryCars.run_ConStGroup(CONSTANTS_DICT)