

# ------------------------------------------------------------------------------------------
# My fancy distance function
# ------------------------------------------------------------------------------------------
struct MulWeightedDistance <: Distances.SemiMetric
end

function (_::MulWeightedDistance)(x::AbstractVector{F}, y::AbstractVector{F}) where {F<:AbstractFloat}
    x_length = sqrt(sum(abs2, x))
    y_length = sqrt(sum(abs2, y))
    # cosine_dist * |x| * |y|
    # = (1 - dot(x, y) / (|x| * |y|)) * |x| * |y|
    # = |x| * |y| - dot(x, y)
    return x_length * y_length - LinearAlgebra.dot(x, y)
end

function Distances.pairwise(md::MulWeightedDistance, X::AbstractMatrix{F}) where {F<:AbstractFloat}
    return Distances.pairwise(md, X, X)
end

function Distances.pairwise(md::MulWeightedDistance, X::AbstractMatrix{F}, Y::AbstractMatrix{F}) where {F<:AbstractFloat}
    # result size is (X_col_n, Y_col_n)
    result = X' * Y
    x_lengths = sqrt.(sum(abs2, X, dims=1))
    y_lengths = sqrt.(sum(abs2, Y, dims=1))
    LoopVectorization.@turbo for y_ind in 1:size(Y, 2), x_ind in 1:size(X, 2)
        result[x_ind, y_ind] = x_lengths[x_ind] * y_lengths[y_ind] - result[x_ind, y_ind]
    end
    return result
end

struct MulNormDistance <: Distances.SemiMetric
end

function (_::MulNormDistance)(x::AbstractVector{F}, y::AbstractVector{F}) where {F<:AbstractFloat}
    x_norm = x .- Statistics.mean(x)
    x_norm ./= Statistics.std(x)
    y_norm = y .- Statistics.mean(y)
    y_norm ./= Statistics.std(y)
    return LinearAlgebra.dot(x_norm, y_norm)
end

function Distances.pairwise(md::MulNormDistance, X::AbstractMatrix{F}) where {F<:AbstractFloat}
    return Distances.pairwise(md, X, X)
end

function Distances.pairwise(md::MulNormDistance, X::AbstractMatrix{F}, Y::AbstractMatrix{F}) where {F<:AbstractFloat}
    # result size is (X_col_n, Y_col_n)
    result = Matrix{F}(undef, size(X, 2), size(Y, 2))
    for y_ind in axes(Y, 2), x_ind in axes(X, 2)
        result[x_ind, y_ind] = md(view(X, :, x_ind), view(Y, :, y_ind))
    end
    return result
end

struct MulDistance <: Distances.SemiMetric
end

function (_::MulDistance)(x::AbstractVector{F}, y::AbstractVector{F}) where {F<:AbstractFloat}
    return LinearAlgebra.dot(x, y)
end

function Distances.pairwise(md::MulDistance, X::AbstractMatrix{F}) where {F<:AbstractFloat}
    return Distances.pairwise(md, X, X)
end

function Distances.pairwise(md::MulDistance, X::AbstractMatrix{F}, Y::AbstractMatrix{F}) where {F<:AbstractFloat}
    return X' * Y
end
