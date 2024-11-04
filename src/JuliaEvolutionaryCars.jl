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

    include("EvolutionaryComputation/ContinuousStatesGrouping/ContinuousStatesGrouping.jl")
    import .ContinuousStatesGrouping

    function run_EvMutPop(CONSTANTS_DICT::Dict{Symbol})
        # Preprocessing data
        final_dict = Dict{Symbol, Any}(
            CONSTANTS_DICT[:Evolutionary_Mutate_Population]...,

            :environment_kwargs => Environment.prepare_environments_kwargs(
                CONSTANTS_DICT[:environment][:universal_kwargs],
                CONSTANTS_DICT[:environment][:changeable_training_kwargs_list]
            ),
            :visualization_kwargs => Dict{Symbol, Any}(CONSTANTS_DICT[:environment][:visualization]),
            :environment_visualization_kwargs => Environment.prepare_environments_kwargs(
                CONSTANTS_DICT[:environment][:universal_kwargs],
                CONSTANTS_DICT[:environment][:changeable_validation_kwargs_list]
            )[1],
            :environment => CONSTANTS_DICT[:environment][:name],
            :neural_network_data => CONSTANTS_DICT[:neural_network]
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

    function run_ConStGroup(CONSTANTS_DICT::Dict{Symbol})
        # Preprocessing data
        final_dict = Dict{Symbol, Any}(
            CONSTANTS_DICT[:ContinuousStatesGrouping]...,

            :environment_kwargs => Environment.prepare_environments_kwargs(
                CONSTANTS_DICT[:environment][:universal_kwargs],
                CONSTANTS_DICT[:environment][:changeable_training_kwargs_list]
            ),
            :visualization_kwargs => Dict{Symbol, Any}(CONSTANTS_DICT[:environment][:visualization]),
            :environment_visualization_kwargs => Environment.prepare_environments_kwargs(
                CONSTANTS_DICT[:environment][:universal_kwargs],
                CONSTANTS_DICT[:environment][:changeable_validation_kwargs_list]
            )[1],
            :environment => CONSTANTS_DICT[:environment][:name],
            :neural_network_data => CONSTANTS_DICT[:neural_network]
        )

        

        # Running the algorithm
        evolutionary_algorithm = ContinuousStatesGrouping.ContinuousStatesGroupingAlgorithm(;final_dict...)
        ContinuousStatesGrouping.run!(evolutionary_algorithm; CONSTANTS_DICT[:run_config]...)
    end

end # module EvolutionaryCarsJulia


# Ogólny problem:
# Jak zrownoleglić obliczeni FIHCA?

# Ogólne pytanie:
# Gdzie się robi zrównoleglenie w P3?

# Pomysł!
# Bierzemy sobie P3, normalne FIHCowanie, potem jeśli przechodzimy do kolejnego poziomu
# to aktualizujemy nasz problem, więc FIHCujemy też go.
# dzięki temu mamy stałe fitnessy poza jednym individualem, który jest i tak FIHCowany, czyli powinien wskoczyć na dobry poziom.

# Pomysł do Evolutionary Mutate Population!
# FIHC - bierzemy jakiegoś individuala, encodujemy stany używając którejś jego warstwy (np. przed ostatniej), grupujemy i FIHCujemy tak jak dotychczas wymyśliłem.
# albo bierzemy te warstwy z 2 individualów, możemy je konkatenować, grupować po tym i robić optimal mixing
# wtedy do obliczenia fitnessu można spróbować uczyć z przeuczać tą sieć
# Pytanie: czy nie będzie problemem optymalizacja losowymi wartościami i gradientowa używana naprzemian?


# przyspieszanie FIHC'abstract
# coś z odległościami między stanami
# jakaś rozmyta funkcja przynależności


# Komar wszystkie ciągłe prace - napisać mail 


# pomysł z zupełnie innej beczki - p3 evolutionary mutate, najpierw jakiś fihc, potem pokolei na jakiejś podstawie genieclust robimy optimalmixing z innymi individualami
# może ten sam mechanizm klasteryzacji itd, ale za


include("constants.jl")
# Run the algorithm
# JuliaEvolutionaryCars.run_EvMutPop(CONSTANTS_DICT)
JuliaEvolutionaryCars.run_StGroupGA(CONSTANTS_DICT)
# JuliaEvolutionaryCars.run_ConStGroup(CONSTANTS_DICT)