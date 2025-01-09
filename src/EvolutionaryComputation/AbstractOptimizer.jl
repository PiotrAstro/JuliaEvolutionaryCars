module AbstractOptimizerModule

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

end # module AbstractOptimizerModule
 