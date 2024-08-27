using MKL

using LinearAlgebra
println("BLAS kernel: $(BLAS.get_config())")
println("Number of BLAS threads: $(BLAS.get_num_threads())")
println("Number of Julia threads: $(Threads.nthreads())")

module JuliaEvolutionaryCars

    export run_EvMutPop

    include("ClusteringHML/ClusteringHML.jl")
    import .ClusteringHML

    include("NeuralNetwork/NeuralNetwork.jl")
    import .NeuralNetwork

    include("Environments/Environment.jl")
    import .Environment

    include("EvolutionaryComputation/EvolutionaryMutatePopulation.jl")
    import .EvolutionaryMutatePopulaiton

    include("EvolutionaryComputation/StatesGroupingGA/StatesGroupingGA.jl")
    import .StatesGroupingGA

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

    function run_StGroupGA(CONSTANTS_DICT::Dict{Symbol})
        # Preprocessing data
        final_dict = Dict{Symbol, Any}(
            CONSTANTS_DICT[:StatesGroupingGA]...,

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
        evolutionary_algorithm = StatesGroupingGA.StatesGroupingGA_Algorithm(;final_dict...)
        StatesGroupingGA.run!(evolutionary_algorithm; CONSTANTS_DICT[:run_config]...)
    end

end # module EvolutionaryCarsJulia



# Ścignąć o tym maila
# O klasteryzacji spytać Jakuba Nalepę
# Zacząć od K-średnich



include("constants.jl")
# Run the algorithm
# JuliaEvolutionaryCars.run_EvMutPop(CONSTANTS_DICT)
JuliaEvolutionaryCars.run_StGroupGA(CONSTANTS_DICT)