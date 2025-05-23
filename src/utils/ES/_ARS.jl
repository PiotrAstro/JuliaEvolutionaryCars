# based on paper: Simple random search provides a competitive approach to reinforcement learning
# absolutely amazing paper!
# https://arxiv.org/pdf/1803.07055
export ARSState

function get_es(::Val{:ARS})
    return ARSState
end

@kwdef mutable struct ARSState{F<:AbstractFloat, ARR<:Array{F}} <: AbstractES
    const maximize::Bool  # if we want to maximize fitness function then true
    const means::ARR
    const child_n::Int
    const best_n::Int
    const sigma::F
    const step_size::F
end

function ARSState(
    means::Array{F},
    ;
    child_n::Int,
    best_n::Int,
    sigma::F=F(0.5f0),
    step_size::F=F(0.5f0),
    maximize::Bool=true,
) where F<:AbstractFloat
    return ARSState(
        maximize=maximize,
        means=Base.copy(means),
        child_n=child_n,
        best_n=best_n,
        sigma=sigma,
        step_size=step_size,
    )
end

function generate_solutions!(ars::ARSState{F, ARR})::Vector{ARR} where {F<:AbstractFloat, ARR<:Array{F}}
    solutions = Vector{ARR}(undef, ars.child_n*2)
    for i in 1:2:(ars.child_n*2)
        random_change = ars.sigma .* randn(F, size(ars.means))
        solutions[i] = ars.means + random_change
        solutions[i + 1] = ars.means - random_change
    end

    return solutions
end

struct _SolutionSet{F, ARR} <: AbstractES
    plus_s::ARR
    minus_s::ARR
    plus_f::F
    minus_f::F
end

function update!(ars::ARSState{F, ARR}, solutions::Vector{ARR}, fitness::Vector{<:AbstractFloat}) where {F<:AbstractFloat, ARR<:Array{F}}
    @assert length(solutions) == ars.child_n*2
    @assert length(fitness) == ars.child_n*2

    solutions_sets = [
        _SolutionSet(solutions[i], solutions[i + 1], fitness[i], fitness[i + 1])
        for i in 1:2:(ars.child_n*2)
    ]

    solutions_taken = sort(solutions_sets, by=x -> max(x.plus_f, x.minus_f), rev=ars.maximize)[1:ars.best_n]
    fitnesses_taken = Vector{F}(undef, ars.best_n * 2)
    for i in 0:ars.best_n-1
        i_sol = i + 1
        i_fitness = i * 2 + 1
        fitnesses_taken[i_fitness] = solutions_taken[i_sol].plus_f
        fitnesses_taken[i_fitness + 1] = solutions_taken[i_sol].minus_f
    end
    std_deviation = Statistics.std(fitnesses_taken)
    means_change = zeros(F, size(ars.means))
    for sol in solutions_taken
        reward = (sol.plus_f - sol.minus_f) * (ars.maximize ? 1 : -1)
        multiplier = ars.step_size / (ars.best_n * std_deviation)
        direction = (sol.plus_s - sol.minus_s) / (2 * ars.sigma)
        means_change .+= (reward * multiplier) .* direction
    end
    ars.means .+= means_change
end

function get_mean(ars::ARSState{F, ARR})::ARR where {F<:AbstractFloat, ARR<:Array{F}}
    return Base.copy(ars.means)
end