module EnvironmentFunctions
    include("Environment.jl")
    include("CarEnvironment.jl")
    using .Environment
    using .CarEnvironment

    export get_environment, prepare_environemnts_kwargs

    "Should return Type{T} where T<:AbstractEnvironment, but it is impossible to write like that"
    function get_environment(name::Symbol) :: Type
        if name == :BasicCarEnvironment
            return BasicCarEnvironment
        else
            throw("Environment not found")
        end
    end

    function prepare_environemnts_kwargs(dict_universal::Dict{Symbol, Any}, dict_changeable::Vector{Dict{Symbol, Any}}) :: Vector{Dict{Symbol, Any}}
        dicts_copy = [deepcopy(dict_universal) for _ in 1:length(dict_changeable)]
        for i in 1:length(dict_changeable)
            for (key, value) in dict_changeable[i]
                dicts_copy[i][key] = value
            end
        end

        return dicts_copy
    end
end