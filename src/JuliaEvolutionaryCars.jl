module JuliaEvolutionaryCars

import DataFrames

export run

include("NeuralNetwork/NeuralNetwork.jl")
import .NeuralNetwork

include("Environments/Environment.jl")
import .Environment

include("utils/utils.jl")
using .Utils

include("EvolutionaryComputation/AbstractOptimizer.jl")
import .AbstractOptimizerModule

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

    optimizer_instance = AbstractOptimizerModule.get_optimizer(optimizer)(;final_dict...)
    data_frame = AbstractOptimizerModule.run!(optimizer_instance; CONSTANTS_DICT[:run_config]...)
    return data_frame
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

# ------------------------------------------------------------------------------------------------------
# links:
# https://arxiv.org/pdf/1810.05691 -> pam fast

# Pomysl na usprawnienie tego co jest obecnie:
# - zrobić lekkiego local searcha wartości przy optimal mixie - czyli np. sprawdzać co by było gdybyśmy wzięli drugą najbardziej popularną odpowiedź, nie tylko tą najbardziej popularną
# - zamiast tłumaczyć z tamtego reprezentanta, można spróbować zapożyczyć w jakiś sposób jego reprezentantów?
#   Albo w jakiś sposób decydować czy powinien on zastąpić mojego czy też nie, albo po prosu ich dokładać, albo usuwać moich n- najbliższych sąsiadów

# inny pomysł - wrócic do motywu posiadania osobnej sieci dekodującej:
# ma to wady w postaci braku determinizmu i potrzebie nauki
# natomiast będziemy to uczyli tylko na exemplarach, więc będzie dość szybkie
# mamy dużą elastyczność, od razu dostajemy rozmyte przynależności
# łatwiej też robić crossovera - puszczamy tych reprezentantów których chcemy połączyć przez drugą sieć i uczymy na wynikach
# potencjalnie można też wziąć reprezentantów z drugiej sieci i uczyć na tych reprezentantach, ale tu trzeba jeszcze przemyśleć, jak ich wybierać i czy zastępować nimi któryś z naszych reprezentantów




# ------------------------------------------------------------------------------------------------------
# kolejny zupełnie inny pomysł xD:
# - przechodzimy na wartości ciągłe
# - każdy indiwidual przechowuje swojego game decodera i matrix wartości docelowych
# - matrix wartości docelowych zmieniamy np. poprzez dodanie do jakiegoś pola 1.0 i dzielenie, żeby suma całości była równa 1.0
# - docelowo można zrobić coś z zapożyczaniem reprezentantów
# - można zrobić wtedy różne ilości klastrów - np. do fihca nbajpierw będziemy dodawali 1.0 do 20 klastrów, poem zrobimy sobie 40 klastrów i będziemy dodawali po 0.5, potem 80 klastrów i będziemy dodawali 0.25
# trochę nadal problem co zrobić z time distance tree



# ------------------------------------------------------------------------------------------------------
# testy local searcha:
# na łatwiejszym środowisku (bez przyspieszania)
# oceniamy z przymatu liczby ewaluacji
# tych ewalucacji może być np. max 10000
# bardzo dużo prób, np. 40?
# na   nclust = 20, 40, 80   -   mval raczej wystarczy 2, ale można spróbować 1
# 1. Dodawanie matrixa randomowego   -   mutation factor (0.1, 0.2, 0.5)   -   norm /sum lub -min/sum   -   rand vs randn
# 2. Dodawanie vectora randomowego do jednego genu   -   mutation factor (0.1, 0.2, 0.5)   -   płaski lub hierarchiczny   -   norm /sum lub -min/sum   -   rand vs randn
# 3. Dyskretny local search,dokładnie jedna akcja w genie ma 1.0, jako FIHC  -  płaski lub hierarchiczny
# 4. Dodadawanie wartości do konkretnego gena (prawdopodobnie jako fihc)   -   adding factor (0.1, 0.2, 0.5)   -   płaski lub hierarchiczny   -   norm /sum lub -min/sum
