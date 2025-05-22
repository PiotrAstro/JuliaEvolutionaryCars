module AbstractOptimizerModule

import ..NeuralNetwork
import ..Environment
using ..Utils

import DataFrames
import Statistics

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

function fitnesses_reduction(method::Symbol, fitnesses::AbstractVector{Float64}) :: Float64
    if method == :mean
        return Statistics.mean(fitnesses)
    elseif method == :median
        return Statistics.median(fitnesses)
    elseif method == :max
        return maximum(fitnesses)
    elseif method == :min
        return minimum(fitnesses)
    elseif method == :sum
        return sum(fitnesses)
    else
        throw(ArgumentError("Unknown fitness reduction method: $method"))
    end
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

include("ContinuousStatesGroupingGA/ContinuousStatesGroupingFIHC.jl")
import .ContinuousStatesGroupingFIHC

include("ContinuousStatesGroupingGA/ContinuousStatesGroupingES.jl")
import .ContinuousStatesGroupingES

include("EncoderOutput/EncoderOutputES.jl")
import .EncoderOutputES

end # module AbstractOptimizerModule
 