# based on The CMA Evolution Strategy: A Tutorial
# https://arxiv.org/pdf/1604.00772
export CMAESState

function get_es(::Val{:CMAES})
    return CMAESState
end

@kwdef mutable struct CMAESState{F<:AbstractFloat, ARR<:Array{F}} <: AbstractES
    const maximize::Bool  # if we want to maximize fitness function then true
    const lambda_n::Int
    const parents_n::Int
    sigma::F
    const means::ARR

    # adaptation
    const c_sigma::F
    const c_c::F
    const c_1::F
    const c_mu::F
    const damping::F

    # weights
    const weights::Vector{F}
    const mueff::F
 
    # changing staff
    const p_c::Vector{F}
    const p_sigma::Vector{F}
    const cov_mat::Matrix{F}
    const invsqrt_cov_mat::Matrix{F}

    # precomputed
    const expected_N::F

    solutions_count::Int = 0
end

function CMAESState(
    ::Type{F},
    dim_size::NTuple,
    ;
    kwargs...,
) where F<:AbstractFloat
    means = zeros(F, dim_size...)
    return CMAESState(
        means,
        kwargs...,
    )
end

function CMAESState(
    means::Array{F},
    ;
    lambda_n::Int = 4 + floor(Int, 3 * log(length(means))),
    parents_n::Int = floor(Int, lambda_n / 2),
    sigma::F=F(0.5f0),
    maximize::Bool=true,
) where F<:AbstractFloat
    means = Base.copy(means)
    dim_number = length(means)

    weights = log(parents_n + F(1/2)) .- log.(1:parents_n)
    weights ./= sum(weights)
    weights = Vector{F}(weights)
    mueffs = F(sum(weights)^2 ./ sum(weights.^2))

    c_sigma=F((mueffs + 2) / (mueffs + dim_number + 5))
    c_c=F((4 + mueffs / dim_number) / (dim_number + 4 + 2 * mueffs / dim_number))
    c_1=F(2 / (mueffs + (dim_number + 1.3)^2))
    c_mu=min(F(1 - c_1), F(2 * (mueffs - 2 + 1/mueffs) / ((dim_number + 2)^2 + mueffs)))
    damping = F(1 + 2*max(zero(F), sqrt((mueffs - 1) / (dim_number + 1)) - 1) + c_sigma)
    expected_n = F(dim_number^0.5 * (1 - 1/(4*dim_number) + 1/(21*dim_number^2)))
    
    return CMAESState(
        maximize=maximize,
        lambda_n=lambda_n,
        parents_n=parents_n,
        sigma=sigma,
        means=means,

        c_sigma=c_sigma,
        c_c=c_c,
        c_1=c_1,
        c_mu=c_mu,
        damping=damping,

        weights=weights,
        mueff=mueffs,

        p_c=zeros(F, dim_number),
        p_sigma=zeros(F, dim_number),
        cov_mat=Matrix{F}(LinearAlgebra.I, dim_number, dim_number),
        invsqrt_cov_mat=Matrix{F}(LinearAlgebra.I, dim_number, dim_number),

        expected_N=expected_n
    )
end

function generate_solutions!(cmaes::CMAESState{F, ARR})::Vector{ARR} where {F<:AbstractFloat, ARR<:Array{F}}
    # Generate solutions
    solutions = [Base.copy(cmaes.means) for _ in 1:cmaes.lambda_n]
    B, D = try
        eigen_result = LinearAlgebra.eigen(LinearAlgebra.Symmetric(cmaes.cov_mat))
        eigen_result.vectors, [sqrt(max(zero(F), F(x))) for x in eigen_result.values]
    catch ex
        throw("Break on eigendecomposition: $ex: $(cmaes.cov_mat)")
    end

    cmaes.invsqrt_cov_mat .= B * LinearAlgebra.Diagonal(D.^-1) * B'
    mul_values = cmaes.sigma .* B * LinearAlgebra.Diagonal(D)
    for solution_id in 1:cmaes.lambda_n
        solution_tmp = solutions[solution_id]
        tmp_result = mul_values * randn(F, length(D), 1)
        for i in eachindex(tmp_result)
            solution_tmp[i] += tmp_result[i]
        end
    end

    return solutions
end

function update!(cmaes::CMAESState{F, ARR}, solutions::Vector{ARR}, fitness::Vector{<:AbstractFloat}) where {F<:AbstractFloat, ARR<:Array{F}}
    @assert length(solutions) == cmaes.lambda_n
    @assert length(fitness) == cmaes.lambda_n
    cmaes.solutions_count += cmaes.lambda_n

    # Sort the solutions by fitness
    sorted_indices = sortperm(fitness, rev=cmaes.maximize)  # I assume the fitness is maximized
    best_indices = sorted_indices[1:cmaes.parents_n]

    # Update means
    mean_old = Base.copy(cmaes.means)
    cmaes.means .= zero(F)
    for (i, parent_idx) in enumerate(best_indices)
        parent = solutions[parent_idx]
        weight = cmaes.weights[i]
        for j in eachindex(parent)
            cmaes.means[j] += weight * parent[j]
        end
    end

    # Update evolution paths
    mean_diff = vec(cmaes.means - mean_old)
    cmaes.p_sigma .= (1-cmaes.c_sigma) .* cmaes.p_sigma + (sqrt(cmaes.c_sigma * (2 - cmaes.c_sigma) * cmaes.mueff) / cmaes.sigma) .* (cmaes.invsqrt_cov_mat * mean_diff)
    threashold_val = 1.4 + 2/(length(mean_diff) + 1)
    norm_p_sigma = LinearAlgebra.norm(cmaes.p_sigma)
    h_sig = (norm_p_sigma / sqrt(1 - (1 - cmaes.c_sigma)^(2 * cmaes.solutions_count / cmaes.lambda_n)) / cmaes.expected_N) < threashold_val
    cmaes.p_c .= (1 - cmaes.c_c) .* cmaes.p_c
    if h_sig
        cmaes.p_c .+= (sqrt(cmaes.c_c * (2 - cmaes.c_c) * cmaes.mueff) / cmaes.sigma) .* mean_diff
    end

    # Update covariance matrix
    parents_influence = Matrix{F}(undef, length(mean_diff), cmaes.parents_n)
    for (i, parent_idx) in enumerate(best_indices)
        parent = solutions[parent_idx]
        mul_val = F(1 / cmaes.sigma)
        for j in eachindex(parent)
            parents_influence[j, i] = (parent[j] - cmaes.means[j]) * mul_val
        end
    end
    cmaes.cov_mat .= (1 - cmaes.c_1 - cmaes.c_mu) .* cmaes.cov_mat +
        cmaes.c_1 .* (cmaes.p_c * cmaes.p_c' + ((!h_sig) * cmaes.c_c * (2-cmaes.c_c)) .* cmaes.cov_mat) +
        cmaes.c_mu * (parents_influence * LinearAlgebra.Diagonal(cmaes.weights) * parents_influence')
    _symmetrize_matrix!(cmaes.cov_mat)

    # Update step size
    cmaes.sigma = cmaes.sigma * exp((cmaes.c_sigma / cmaes.damping) * (norm_p_sigma / cmaes.expected_N - 1))
end

function get_mean(cmaes::CMAESState{F, ARR})::ARR where {F<:AbstractFloat, ARR<:Array{F}}
    return Base.copy(cmaes.means)
end