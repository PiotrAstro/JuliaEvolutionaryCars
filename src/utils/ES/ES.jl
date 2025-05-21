module ES

import Statistics
import LinearAlgebra
import LoopVectorization
import Random

export generate_solutions!, update!, get_mean, get_es, AbstractES

function get_es(es::Symbol)
    return get_es(Val(es))
end

# --------------------------------------------------------------------------------
# Abstract evolutionary stragtegy type
abstract type AbstractES end

function get_es(es::Val)
    throw(ArgumentError("get_es not implemented for $(es)"))
end

"""
Returns vector of Arrays
"""
function generate_solutions!(es::AbstractES)
    throw(ArgumentError("generate_solutions! not implemented for $(typeof(es))"))
end

"""
updates internal things of evolutionary strategy
"""
function update!(es::AbstractES, solutions::Vector{<:Array}, fitness::Vector{<:AbstractFloat})
    throw(ArgumentError("update! not implemented for $(typeof(es))"))
end

"""
Get mean (central, memorized) array of evolutionary strategy
"""
function get_mean(es::AbstractES)
    throw(ArgumentError("get_mean not implemented for $(typeof(es))"))
end


# --------------------------------------------------------------------------------
# utils
function _symmetrize_matrix!(mat::Matrix)
    @assert size(mat, 1) == size(mat, 2)
    for i in axes(mat, 2)
        @fastmath @inbounds @simd for k in 1:(i-1)
            mat[i, k] = mat[k, i]
        end
    end
end

# --------------------------------------------------------------------------------
# includes
include("_CMAES.jl")


#---------------------------------------------------------------------------------
# testing
# used for plotting gif's and visually inspecting the results


# import Plots

# function ackley(x::Vector)
#     n = length(x)
#     sum1 = sum(x.^2)
#     sum2 = sum(cos.(2π * x))
    
#     term1 = -20.0 * exp(-0.2 * sqrt(sum1 / n))
#     term2 = -exp(sum2 / n)
    
#     return -(term1 + term2 + 20.0 + exp(1.0)) # it will be maximized
# end

# function rastrigin(x::Vector)
#     n = length(x)
#     A = 10.0
#     return -(A * n + sum(x.^2 - A * cos.(2π * x))) # it will be maximized
# end

# function dropwave(v::Vector)
#     x, y = v[1], v[2]
#     numerator = 1 + cos(12 * sqrt(x^2 + y^2))
#     denominator = 0.5 * (x^2 + y^2) + 2
    
#     return -numerator / denominator * -1 # it will be maximized
# end

# function crossintray(v::Vector)
#     x, y = v[1], v[2]
#     term = abs(sin(x) * sin(y) * exp(abs(100 - sqrt(x^2 + y^2)/π))) + 1
    
#     return -0.0001 * (term^0.1) * -1 # it will be maximized
# end

# function schwefel(x::Vector)
#     n = length(x)
#     # x = copy(x)
#     # x .*= 100 # I want to get more managable sizes
#     sum_term = sum(x[i] * sin(sqrt(abs(x[i]))) for i in 1:n)
#     official_value = 418.9829 * n - sum_term
#     return official_value * -1 # it will be maximized
# end


# function animate(f, es::AbstractES, iter, range_val)
#     x_range = range(-range_val, range_val, length=300)
#     y_range = range(-range_val, range_val, length=300)
#     z_data = [f([x, y]) for y in y_range, x in x_range]
    
#     anim = Plots.@animate for i in 1:iter
#         means = get_mean(es)
#         solutions = generate_solutions!(es)
#         fitness = f.(solutions)
#         update!(es, solutions, fitness)
        
#         # Create heatmap with contour lines
#         p = Plots.heatmap(
#             x_range, y_range, z_data,
#             c=:viridis,
#             colorbar=true,
#             xlims=(-range_val, range_val),
#             ylims=(-range_val, range_val),
#             aspect_ratio=1,
#             xlabel="x",
#             ylabel="y",
#             title="$(f) - Iteration $i"
#         )
        
#         # Add population points
#         Plots.scatter!(
#             p,
#             [sol[1] for sol in solutions],
#             [sol[2] for sol in solutions],
#             markersize=4,
#             color=:red,
#             markerstrokewidth=0,
#             alpha=0.8,
#             label=false
#         )

#         # Add mean point
#         Plots.scatter!(
#             p,
#             [means[1]], [means[2]],
#             color=:pink,
#             markersize=3,
#             markerstrokewidth=1,
#             markerstrokecolor=:black,
#             label=false
#         )
#     end
    
#     return anim
# end

# """
# It runs simple evolutionary strategy and saves gif
# example:
#     `gif_save(:CMAES, (;sigma=3f0), :ackley, "data/test.gif")`
# """
# function gif_save(es::Symbol, kwargs, fun::Symbol, filename::String; start=[6f0, 6f0], iter=30, range_val=8, fps=5)
#     if fun == :ackley
#         f = ackley
#     elseif fun == :rastrigin
#         f = rastrigin
#     elseif fun == :dropwave
#         f = dropwave
#     elseif fun == :crossintray
#         f = crossintray
#     elseif fun == :schwefel
#         f = schwefel
#     else
#         throw(ArgumentError("Function $(fun) not implemented"))
#     end

#     es = get_es(es)(start; kwargs...)
#     anim = animate(f, es, iter, range_val)
#     Plots.gif(anim, filename, fps=fps)
# end

end # module

# import .ES
# ES.gif_save(:CMAES, (;sigma=1f0, lambda_n=10), :rastrigin, "data/test.gif")