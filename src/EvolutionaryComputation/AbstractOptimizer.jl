module AbstractOptimizerModule

import DataFrames

abstract type AbstractOptimizer end

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
 