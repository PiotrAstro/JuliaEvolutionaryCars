# based on The CMA Evolution Strategy: A Tutorial
# https://arxiv.org/pdf/1604.00772
export DEState

function get_es(::Val{:DE})
    return DEState
end

@kwdef mutable struct DEState{F<:AbstractFloat, ARR<:Array{F}} <: AbstractES
    const maximize::Bool  # if we want to maximize fitness function then true
    const norm_std::Bool
    const fitnesses::Vector{F}
    const parents::Vector{ARR}
    const base_mode::Symbol
    const f_param::F
    const cross_param::F
    const cross_per_symbol::Symbol
end

function DEState(
    means::Array{F},
    ;
    parents_n::Int,
    cross_per_symbol::Symbol=:col,
    base_mode::Symbol=:rand,
    sigma::F=F(0.5f0),
    f_param::F=F(0.5f0),
    cross_param::F=F(0.5f0),
    maximize::Bool=true,
    norm_std::Bool=false,
) where F<:AbstractFloat
    parents = [means + sigma .* randn(F, size(means)) for _ in 1:parents_n]

    if norm_std
        for solution in parents
            solution .-= Statistics.mean(solution)
            solution ./= Statistics.std(solution)
        end
    end

    base_fitness = maximize ? typemin(F) : typemax(F)
    fitnesses = [base_fitness for _ in 1:parents_n]
    return DEState(
        maximize=maximize,
        fitnesses=fitnesses,
        parents=parents,
        base_mode=base_mode,
        f_param=f_param,
        cross_param=cross_param,
        norm_std=norm_std,
        cross_per_symbol=cross_per_symbol,
    )
end

function generate_solutions!(de::DEState{F, ARR})::Vector{ARR} where {F<:AbstractFloat, ARR<:Array{F}}
    # Generate solutions
    solutions = [Base.copy(de.parents[1]) for _ in 1:length(de.parents)]
    best_solution = de.parents[argmax(de.fitnesses)]
    for i in 1:length(de.parents)
        # Select three random parents
        p1, p2, p3 = Random.randperm(length(de.parents))[1:3]

        # Generate new solution based on the selected parents
        if de.base_mode == :rand
            base = de.parents[p1]
        elseif de.base_mode == :best
            base = best_solution
        else
            error("Unknown base mode: $(de.base_mode)")
        end

        applied_all_cross = base + de.f_param .* (de.parents[p2] - de.parents[p3])

        if de.cross_per_symbol == :col
            for col in axes(applied_all_cross, 2)
                if rand() < de.cross_param
                    solutions[i][:, col] = applied_all_cross[:, col]
                end
            end
        elseif de.cross_per_symbol == :row
            for row in axes(applied_all_cross, 1)
                if rand() < de.cross_param
                    solutions[i][row, :] = applied_all_cross[row, :]
                end
            end
        elseif de.cross_per_symbol == :all
            for j in eachindex(solutions[i])
                if rand() < de.cross_param
                    solutions[i][j] = applied_all_cross[j]
                end
            end
        else
            error("Unknown cross_per_symbol mode: $(de.cross_per_symbol)")
        end
    end

    if de.norm_std
        for solution in solutions
            solution .-= Statistics.mean(solution)
            solution ./= Statistics.std(solution)
        end
    end

    return solutions
end

function update!(de::DEState{F, ARR}, solutions::Vector{ARR}, fitness::Vector{<:AbstractFloat}) where {F<:AbstractFloat, ARR<:Array{F}}
    @assert length(solutions) == length(de.fitnesses)
    @assert length(fitness) == length(de.fitnesses)
    func = de.maximize ? (>) : (<)
    for i in eachindex(de.parents)
        if func(fitness[i], de.fitnesses[i])
            de.fitnesses[i] = fitness[i]
            de.parents[i] = Base.copy(solutions[i])
        end
    end
end

function get_mean(de::DEState{F, ARR})::ARR where {F<:AbstractFloat, ARR<:Array{F}}
    mean = zeros(F, size(de.parents[1]))
    for i in eachindex(de.parents)
        mean += de.parents[i]
    end
    mean ./= length(de.parents)
    return mean
end