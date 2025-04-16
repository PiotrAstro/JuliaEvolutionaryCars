module AbstractOptimizerModule

import ..NeuralNetwork
import ..Environment
using ..Utils

import DataFrames

function get_optimizer(optimizer::Symbol)
    return get_optimizer(Val(optimizer))
end


# Interface:

abstract type AbstractOptimizer end

function get_optimizer(optimizer::Val{T}) where T
    throw("not implemented")
end

function run!(
        optimizer::AbstractOptimizer;
        max_generations,
        max_evaluations,
        log,
        visualize_each_n_epochs
    ) :: DataFrames.DataFrame
    throw("not implemented")
end

include("EvolutionaryMutatePopulation/EvolutionaryMutatePopulation.jl")
import .EvolutionaryMutatePopulaiton

# this one is currently depracated
# include("StatesGroupingGA/StatesGroupingGA.jl")
# import .StatesGroupingGA

# this one is currently depracated
# include("ContinuousStatesGroupingGA/ContinuousStatesGroupingP3.jl")
# import .ContinuousStatesGroupingP3

# this one is also depracated!
# include("ContinuousStatesGroupingGA/ContinuousStatesGroupingSimpleGA.jl")
# import .ContinuousStatesGroupingSimpleGA

include("ContinuousStatesGroupingGA/ContinuousStatesGroupingDE.jl")
import .ContinuousStatesGroupingDE

end # module AbstractOptimizerModule
 