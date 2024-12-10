module JuliaEvolutionaryCars

import DataFrames

export run

include("EvolutionaryComputation/AbstractOptimizer.jl")
import .AbstractOptimizerModule

include("NeuralNetwork/NeuralNetwork.jl")
import .NeuralNetwork

include("Environments/Environment.jl")
import .Environment

include("EvolutionaryComputation/EvolutionaryMutatePopulation/EvolutionaryMutatePopulation.jl")
import .EvolutionaryMutatePopulaiton

include("EvolutionaryComputation/StatesGroupingGA/StatesGroupingGA.jl")
import .StatesGroupingGA

include("EvolutionaryComputation/ContinuousStatesGrouping/ContinuousStatesGrouping.jl")
import .ContinuousStatesGrouping

function run(optimizer::Symbol, CONSTANTS_DICT::Dict{Symbol}) :: DataFrames.DataFrame
    # Preprocessing data
    final_dict = Dict{Symbol, Any}(
        CONSTANTS_DICT[optimizer]...,

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

    optimizer_instance = get_optimizer(optimizer)(;final_dict...)
    data_frame = AbstractOptimizerModule.run!(optimizer_instance; CONSTANTS_DICT[:run_config]...)
    return data_frame
end

function get_optimizer(optimizer::Symbol)
    if optimizer == :StatesGroupingGA
        return StatesGroupingGA.StatesGroupingGA_Algorithm
    elseif optimizer == :EvolutionaryMutatePopulation
        return EvolutionaryMutatePopulaiton.EvolutionaryMutatePopulationAlgorithm
    elseif optimizer == :ContinuousStatesGrouping
        return ContinuousStatesGrouping.ContinuousStatesGroupingAlgorithm
    else
        throw("Optimizer not found")
    end
end

end # module EvolutionaryCarsJulia


# ------------------------------------------------------------------------------------------------------
# pomysły co zrobić

# co można zrobić?
# - przyspieszenie fihca i crossovera - jeśli środowisko udostepnia możliwość zapisania stanu środowiska,
#   to możemy z nowym indiwidualem najpierw sprawdzić po kolei odpowiedzi na stany i zobaczyć, gdzie by się to dopiero zmieniło i puścić symulacje od tego miejsca

# modyfikacja obecnego P3
# - przy obecnym pomyśle każdy poziom powinien mieć własne drzewo odległości czasowej, cały czas aktualizowane
# - powinienem sprawdzić, co wpływa na sukces - inna liczba branych pod uwagę okolicznych klastrów (w distance based classificator),
#   inna liczba klastrów, brak fihca na każdym poziomie, drzewo podobieństwa stanów zamiast drzewa odległości czasowej
# - spróbować optimal mixing z faktycznie jednym randomowym individualem, a nie sprawdzanie ze wszystkimi i wybór najlepszego


# pomysł na "docelową" modyfikację P3:
# - mam generację kilku/kilkunastu individualów, które razem się wspinają, individuale te mają nie zmieniającego się environment wrappera (i w ramach generacji tego samego)
# - drzewo odległości czasowej tworzymy gdy generacja ma zacząć się mixować z jakimś poziomem (czyli dla każdego poziomu i dla każdej generacji osobne drzewo)
# - wtedy do mixowania individuale mogą mieć zupełnie inne environment wrappery, więc byśmy po prostu tłumaczyli za każdym razem rozwiązania z jednego na drugi system
#   (może gdzieś ciągłe pomogą?), może to całkowicie popsuć wyniki, ale może też pomóc pozwalając na "inne spojrzenie" na problem
# - w takim modelu można zrównoleglać obliczenia w ramach każdej generacji, ale również na każdym poziomie w danym momencie może działać inna generacja
# - jednocześnie cały czas działamy na nowych i aktualizowanych environment wrapperach
#   (nowe generacje dostaną environment wrappery nauczone na generacji, która się właśnie zakończyły, lub tych individuali z najlepszymi wynikami, albo też jakiś mix różnych individuali)
# - pomysł jeśli się to sprawdzi - można zrobić adaptacyjną metodę ustalania metaparametrów dla generacji (np. liczba klastrów, architektura encodera, parametry nauki autoencodera itd.)


# ---------------------------------------------------------------------------------------------------------------------
# O czym powiedzieć i w jakiej kolejności:

# - Czemu pisalem maila, co wtedy nie działało
# - Przyspieszenie, algorytm rozmytej przynależności
#

# ---------------------------------------------------------------------------------------------------------------------
# Notatki ze spotkania

# zobaczyć partition crossover - czyli z DSM wywlamy geny, które są te same i budujemy drzewo tylko dla różnych\
# współczynniki walsha do tworzenia surogatów - może coś z tego wyjść

# ---------------------------------------------------------------------------------------------------------------------
# Testy co zbadać
