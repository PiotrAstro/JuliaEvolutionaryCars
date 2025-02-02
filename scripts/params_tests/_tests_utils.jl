
"""
It will remove all cases that are already done from the special_dicts_with_cases in a given directory, it is done so that when something interupts calculations, we are able to rerun the script and it will not run the same cases again.
special dicts should be:
special_dicts_with_cases::Vector{Tuple{Symbol, <:Dict{Symbol}, <:Dict{Symbol}, Int}}

"""
function consider_done_cases(special_dicts_with_cases::Vector, save_dir::String) 
    files_in_save_dir = readdir(save_dir)
    log_text = "Checking for existing test cases in $save_dir"
    will_run = trues(length(special_dicts_with_cases))
    for (i, (optimiser, special_dict, case_index)) in enumerate(special_dicts_with_cases)
        (optimiser, special_dict, case_indexes) = special_dicts_with_cases[i]
        save_n = save_name(optimiser, special_dict, case_index)
        if save_n in files_in_save_dir
            log_text = log_text * "\ntest case existing, will not run: $save_n"
            will_run[i] = false
        end
    end
    Logging.@info log_text

    return special_dicts_with_cases[will_run]
end


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

function create_all_special_dicts_with_cases(special_dicts, cases_number::Int)
    special_dicts_with_cases = vec([
        (optimizer, special_dict, case)
        for (optimizer, special_dict) in special_dicts, case in 1:cases_number
    ])
    return special_dicts_with_cases
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
function run_one_test!(
        optimizer::Symbol,
        special_dict::Dict{Symbol, <:Any},
        dict_config_copied::Dict{Symbol, <:Any},
        case_index::Int,
        worker_id::Int,
    )
    save_n = save_name(optimizer, special_dict, case_index)
    Logging.@info "Worker $(worker_id) Running test with config:\n" * save_n

    change_dict_value!(dict_config_copied, special_dict)

    data_frame_result = JuliaEvolutionaryCars.run(
        optimizer,
        dict_config_copied
    )

    buf = IOBuffer()
    CSV.write(buf, data_frame_result)
    csv_string = String(take!(buf))

    # Save the results to a file
    message = "Worker $(worker_id) Finished with config:\n" * save_n
    return save_n, csv_string, message
end

function remote_run(one_special_dict_with_case, constants_dict::Dict, task_id::Int, remote_channel::Distributed.RemoteChannel)
    Logging.global_logger(CustomLoggers.RemoteLogger())
    optimizer, special_dict, case = one_special_dict_with_case
    config_copy = deepcopy(constants_dict)

    try
        save_n, remote_string, message = run_one_test!(optimizer, special_dict, config_copy, case, Distributed.myid())
        remote_result = RemoteResult(remote_string, save_n, :success, message, Distributed.myid(), task_id)
        Logging.@info "workerid $(Distributed.myid()) finished with success, sending results to main worker..."
        put!(remote_channel, remote_result)
    catch e
        message = "workerid $(Distributed.myid()) failed with error: $e"
        remote_result = RemoteResult("", save_name(optimizer, special_dict, case), :error, message, Distributed.myid(), task_id)
        Logging.@error message * " sending result to main worker..."
        put!(remote_channel, remote_result)
    end
end

struct RemoteResult
    csv_string::String
    save_name::String
    message_symbol::Symbol
    message::String
    worker_id::Int
    task_id::Int
end

struct FinalResultLog
    save_name::String
    success::Bool
    message::String
end

# I should try not sendind df, and only smaller data, maybe this is the reason?
# function call_main_save_results(df::DataFrames.DataFrame, save_main_host_dir::String, save_name::String, caller_id::Int)
function main_save_results(remote_result::RemoteResult, save_main_host_dir::String) :: FinalResultLog
    
    if remote_result.message_symbol == :error
        Logging.@error "Main worker received error from worker $(remote_result.worker_id): $(remote_result.message)"
        return FinalResultLog(remote_result.save_name, false, remote_result.message)
    elseif remote_result.message_symbol == :success
        try
            save_path = joinpath(save_main_host_dir, remote_result.save_name)
            open(save_path, "w") do io
                write(io, remote_result.csv_string)
            end
            Logging.@info "Main worker finished saving results from pid: $(remote_result.worker_id) to file: $(remote_result.save_name)"
            return FinalResultLog(remote_result.save_name, true, remote_result.message)
        catch e
            message = "Main worker failed to save results from pid: $(remote_result.worker_id) to file: $(remote_result.save_name)\n$e"
            Logging.@error message
            return FinalResultLog(remote_result.save_name, false, message)
        end
    else 
        message = "Main worker received unknown message from worker: $(remote_result.worker_id)"
        Logging.@error message
        return FinalResultLog(remote_result.save_name, false, message)
    end
end


function construct_text_from_final_results(final_results::Vector{FinalResultLog}) :: String
    texts = []
    for final_result in final_results
        push!(texts, (final_result.success ? "success" : "failed") * "  ->  " * final_result.save_name * (final_result.success ? "" : "  ->  " * final_result.message))
    end

    text_log = "Result:\n" * join(texts, "\n")
    return text_log
end

function run_channel_controller!(
        remote_channel::Distributed.RemoteChannel,
        result_info::Vector{FinalResultLog},
        progress_meter::ProgressMeter.Progress,
        save_main_host_dir::String,
        save_result_path::String,
    )

    while true
        remote_result = take!(remote_channel)
        if remote_result.message_symbol == :stop
            break
        end
        final_result = main_save_results(remote_result, save_main_host_dir)
        result_info[remote_result.task_id] = final_result

        try
            ProgressMeter.next!(progress_meter)
            open(save_result_path, "w") do io
                write(io, construct_text_from_final_results(result_info))
            end
        catch e
            Logging.@error "Main worker error (just handling additional staff, not computation error): $save_result_path\n$e"
        end
    end
end