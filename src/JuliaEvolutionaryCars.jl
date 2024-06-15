module JuliaEvolutionaryCars
    export run_EvMutPop

    include("neural_network/NeuralNetwork.jl")
    import .NeuralNetwork

    include("environments/Environment.jl")
    import .Environment

    include("evolutionary_computation/EvolutionaryMutatePopulation.jl")
    import .EvolutionaryMutatePopulaiton

    function run_EvMutPop(CONSTANTS_DICT::Dict{Symbol})
        # Preprocessing data
        final_dict = Dict{Symbol, Any}(
            CONSTANTS_DICT[:Evolutionary_Mutate_Population]...,

            :neural_network_data => CONSTANTS_DICT[:neural_network],
            :environment_kwargs => Environment.prepare_environments_kwargs(
                CONSTANTS_DICT[:environment][:universal_kwargs],
                CONSTANTS_DICT[:environment][:changeable_training_kwargs_list]
            ),
            :visualization_kwargs => Dict{Symbol, Any}(CONSTANTS_DICT[:environment][:visualization]),
            :environment_visualization_kwargs => Environment.prepare_environments_kwargs(
                CONSTANTS_DICT[:environment][:universal_kwargs],
                CONSTANTS_DICT[:environment][:changeable_validation_kwargs_list]
            )[1],
            :environment => CONSTANTS_DICT[:environment][:name]
        )

        

        # Running the algorithm
        evolutionary_algorithm = EvolutionaryMutatePopulaiton.EvolutionaryMutatePopulationAlgorithm(;final_dict...)
        EvolutionaryMutatePopulaiton.run!(evolutionary_algorithm; CONSTANTS_DICT[:run_config]...)
    end

end # module EvolutionaryCarsJulia

include("constants.jl")
# Run the algorithm
JuliaEvolutionaryCars.run_EvMutPop(CONSTANTS_DICT)