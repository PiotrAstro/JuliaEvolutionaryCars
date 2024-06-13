include("constants.jl")
include("evolutionary_computation/EvolutionaryMutatePopulation.jl")
include("environments/EnvironmentFunctions.jl")

using .EnvironmentFunctions
using .EvolutionaryMutatePopulaiton

# Preprocessing data
final_dict = Dict{Symbol, Any}(copy(CONSTANTS_DICT[:Evolutionary_Mutate_Population]))
final_dict[:neural_network_data] = CONSTANTS_DICT[:neural_network]
final_dict[:environment_kwargs] = prepare_environemnts_kwargs(
    CONSTANTS_DICT[:environment][:universal_kwargs],
    CONSTANTS_DICT[:environment][:changeable_training_kwargs_list]
)
final_dict[:visualization_kwargs] = Dict{Symbol, Any}(CONSTANTS_DICT[:environment][:visualization])
final_dict[:environment_visualization_kwargs] = prepare_environemnts_kwargs(
    CONSTANTS_DICT[:environment][:universal_kwargs],
    CONSTANTS_DICT[:environment][:changeable_validation_kwargs_list]
)[1]
final_dict[:environment] = CONSTANTS_DICT[:environment][:name]


# Running the algorithm
evolutionary_algorithm = EvolutionaryMutatePopulationAlgorithm(;final_dict...)
run!(evolutionary_algorithm)
