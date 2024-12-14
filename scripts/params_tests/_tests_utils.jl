
"""
Finds all lists in a dictionary.
:param dict_to_search: The dictionary to search
:return: A vector of tuples, where each tuple contains a vector of keys and the list found

input
"Evolutionary_Mutate_Population": {
   "mutation_factor": [0.01, 0.05, 0.1, 0.2],
   "mutation_threshold": [None],
},

return 
 :: Vector{Tuple{Vector{Symbol}, Vector{<:Any}}}
[
    (
        ["Evolutionary_Mutate_Population", "mutation_factor"],
        [0.01, 0.05, 0.1, 0.2]
    ),
    (
        ["Evolutionary_Mutate_Population", "mutation_threshold"],
        [None]
    )
]

"""
function find_lists_in_dict(dict_to_search::Dict{Symbol, <:Any}) :: Vector{Tuple{Vector{Symbol}, Vector{<:Any}}}
    lists_keys_values = []

    for (key, value) in dict_to_search
        if isa(value, Vector)  # Check if the value is a list (Vector in Julia)
            push!(lists_keys_values, ([key], value))
        elseif isa(value, Dict{Symbol, <:Any})  # Check if the value is a nested dictionary
            tmp_lists_keys_values = find_lists_in_dict(value)
            for (key_list, value_list) in tmp_lists_keys_values
                push!(lists_keys_values, ([key, key_list...], value_list))
            end
        end
    end

    return lists_keys_values
end


"""
Constructs special dictionaries from given keys_values_to_add
so it makes all possible combinations of the values in the lists.

input:
keys_values_to_add = [
    (
        ["Evolutionary_Mutate_Population", "mutation_factor"],
        [0.01, 0.05, 0.1, 0.2]
    ),
    (
        ["Evolutionary_Mutate_Population", "mutation_threshold"],
        [None]
    )
]

so_far_dict = Dict{Symbol, <:Any}() -- current changes

return:
 :: Vector{Dict{Symbol, <:Any}}
[
    Dict{Symbol, <:Any}(
        "Evolutionary_Mutate_Population": {
            "mutation_factor": 0.01,
            "mutation_threshold": None,
        },
    ),
    Dict{Symbol, <:Any}(
        "Evolutionary_Mutate_Population": {
            "mutation_factor": 0.05,
            "mutation_threshold": None,
        },
    ),
]
"""
function construct_special_dicts(keys_values_to_add::Vector{Tuple{Vector{Symbol}, Vector{<:Any}}}, so_far_dict::Dict{Symbol, <:Any}) :: Vector{Dict{Symbol, <:Any}}
    special_dicts = Vector{Dict{Symbol, <:Any}}()

    if isempty(keys_values_to_add)
        push!(special_dicts, so_far_dict)
    else
        key_list, value_list = keys_values_to_add[1]

        for value in value_list
            new_dict = Base.deepcopy(so_far_dict)
            tmp_dict = new_dict

            for key in key_list[1:end-1]
                if !haskey(tmp_dict, key)
                    tmp_dict[key] = Dict{Symbol, Any}()
                end
                tmp_dict = tmp_dict[key]
            end

            tmp_dict[key_list[end]] = value
            append!(special_dicts, construct_special_dicts(keys_values_to_add[2:end], new_dict))
        end
    end

    return special_dicts
end



"""
Creates all special dictionaries from the tested_dicts_list
input:
[
    (
        :Evolutionary_Mutate_Population,
        Dict{Symbol, <:Any}(
            "Evolutionary_Mutate_Population": {
                "mutation_factor": [0.01, 0.05, 0.1, 0.2],
                "mutation_threshold": [None],
            },
        ),
    ),
    (
        :Evolutionary_Mutate_Population,
        Dict{Symbol, <:Any}(
            "Evolutionary_Mutate_Population": {
                "mutation_factor": [0.03, 0.1, 0.3],
                "mutation_threshold": [0.03, 0.1, 0.3],
            },
        ),
    )
]

output:
 :: Vector{Tuple{Symbol, Dict{Symbol, <:Any}}}
[
    (
        :Evolutionary_Mutate_Population, 
        Dict{Symbol, <:Any}(
            "Evolutionary_Mutate_Population": {
                "mutation_factor": 0.01,
                "mutation_threshold": None,
            },
        ),
    )
    (
        :Evolutionary_Mutate_Population,
        Dict{Symbol, <:Any}(
            "Evolutionary_Mutate_Population": {
                "mutation_factor": 0.05,
                "mutation_threshold": 0.03
            },
        ),
    )
    ...
]
"""
function create_all_special_dicts(tested_dicts_list::Vector{<:Tuple{Symbol, <:Dict{Symbol, <:Any}}}) :: Vector{Tuple{Symbol, Dict{Symbol, <:Any}}}
    all_special_dicts = Vector{Tuple{Symbol, Dict{Symbol, Any}}}()

    for (optimizer, tested_dict) in tested_dicts_list
        lists_keys_values = find_lists_in_dict(tested_dict)
        special_dicts = construct_special_dicts(lists_keys_values, Dict{Symbol, Any}())
        append!(all_special_dicts, [(optimizer, special_dict) for special_dict in special_dicts])
    end

    return all_special_dicts
end



"""
Changes values in destination_dict to values from source_dict.
destination is e..g. copy of onfig file, source is one of the special dicts
"""
function change_dict_value!(destination_dict::Dict{Symbol, <:Any}, source_dict::Dict{Symbol, Any})
    for (key, value) in source_dict
        if isa(value, Dict)
            change_dict_value!(destination_dict[key], value)
        else
            destination_dict[key] = value
        end
    end
end

"""
Function to convert a dictionary to a string representation.
input:
"Evolutionary_Mutate_Population": {
        "mutation_factor": [0.01, 0.05, 0.1, 0.2],
        "mutation_threshold": [None],
    },
),

output:
"(EvoMutPop=(MutFac=0.01_MutThr=None))"

"""
function dict_to_name(dict_to_use::Dict{Symbol, <:Any})::String
    name = "("
    for (key, value) in dict_to_use
        if isa(value, Dict)
            name *= shorten_name(string(key)) * "=" * dict_to_name(value)
        else
            name *= shorten_name(string(key)) * "=" * string(value)
        end
        name *= "_"
    end
    name = chop(name, tail=1) * ")"  # Remove the trailing underscore and close the parenthesis
    return name
end

#check if it is camel case or snake case and split it
function split_name(input::String)
    # First, handle snake_case by splitting on underscores
    if contains(input, "_")
        return split(input, "_")
    end
    
    # For camelCase, split by uppercase letters
    words = String[]
    current_word = input[1:1]
    for char in input[2:end]
        if isuppercase(char)
            push!(words, current_word)
            current_word = string(char)
        else
            current_word *= string(char)
        end
    end
    push!(words, current_word)
    
    return words
end

# Function to shorten names of variables
function shorten_name(name::String, cut_length::Int=3)::String
    words = split_name(name)
    for i in eachindex(words)
        if length(words[i]) > cut_length
            words[i] = words[i][1:cut_length]
        end
        words[i] = uppercasefirst(words[i])
    end
    return join(words)
end

# Function to generate a filename from a dictionary and a case index
function save_name(optimizer::Symbol, dict_to_use::Dict{Symbol, <:Any}, case_index::Int; extension::String=".csv")::String
    case_index_string = lpad(string(case_index), 2, "0")
    optimizer_name = shorten_name(string(optimizer))
    return "logs_opt=" * optimizer_name * "_" * dict_to_name(dict_to_use) * "__case" * case_index_string * extension
end

"""
It will run one test with the given special_dict and dict_config.
special dict should be taken from create_all_special_dicts
You should copy dict_config_copied yourself before passing to the function

input:
special_dict = Dict{Symbol, <:Any}(
    "Evolutionary_Mutate_Population": {
        "mutation_factor": 0.01,
        "mutation_threshold": None,
    },
)

dict_config_copied - it should be copied before!
"""
function run_one_test(optimizer::Symbol, special_dict::Dict{Symbol, <:Any}, dict_config_copied::Dict{Symbol, <:Any}, case_index::Int, save_dir)
    save_n = save_name(optimizer, special_dict, case_index)
    Logging.@info "Running test with config: " save_n

    change_dict_value!(dict_config_copied, special_dict)

    data_frame_result = JuliaEvolutionaryCars.run(
        optimizer,
        dict_config_copied
    )

    # Save the results to a file
    CSV.write(joinpath(save_dir, save_n), data_frame_result)
    Logging.@info "Finished with config: " save_n
end
