
import Plots
using Plots.PlotMeasures
import DataFrames
import CSV
import Statistics


# ------------------------------------------------------------------------------------------------
# My params


TEST_DIR = joinpath("log", "parameters_tests_2025-01-09_23-17-50")
RESULTS_DIR = joinpath(TEST_DIR, "results")
ANALYSIS_DIR = joinpath(TEST_DIR, "analysis")

TEST_POSTFIX = ".csv"  # will be removed from plot entries
TEST_PREFIX = "logs_opt=StaGroGA_"  # will be removed from plot entries

Y_LABEL = :best_fitness
X_LABEL = :generation
LINE_METHOD = :mean  # :all, :mean, :median, :max, :min
SHOW_STD = false  # whether to show std ribbon, doesnt matter for :all

GROUPS = []  # By default [], so no GROUPS
GROUPS = ["NClu"]  # could be e.g. ["NClu", "MmdWei"] it will create groups for each combination of these, if entry doesnt have any of these, it will be a group on its own
GROUPS_IN_LEGEND = :all  # :all - different colours in groups, :col1 - one colour in groups, :col1_ent1 - one colour in groups and one entry in legend

# will stay in the plot entries, used for filtering
# TEST_INFIX_LIST = ["40", ("30", "!50")]  ->  contains("40") && (contains("30") || !contains("50"))
# usually you should use it like this TEST_INFIX_LIST = ["(MmdWei=0.0)"] 
# if you add ! as the first string index, it means not this one, e.g. TEST_INFIX_LIST = ["!40", "!50"] -> !contains("40") && !contains("50")
TEST_INFIX_LIST = []


# ------------------------------------------------------------------------------------------------
# Utils
DISTINT_COLOURS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Lime
    "#17becf",  # Cyan
    "#ff9896",  # Light Red
    "#98df8a",  # Light Green
    "#aec7e8",  # Light Blue
    "#ffbb78",  # Light Orange
    "#c5b0d5"   # Light Purple
]
PLOT_SIZE = (1500, 1000)
PLOT_MARGIN = 10mm
PLOT_TOP_MARGIN = 15mm
LINE_WIDTH = 3
RIBBON_FILL_ALPHA = 0.1

function get_name_string(file_name::String) :: String
    remove_prefix = replace(file_name, TEST_PREFIX => "")
    remove_case_extension = split(remove_prefix, "__case")[1]
    if startswith(remove_case_extension, "(")
        remove_case_extension = remove_case_extension[2:end]
    end
    if endswith(remove_case_extension, ")")
        remove_case_extension = remove_case_extension[1:end-1]
    end

    return remove_case_extension
end

"""
It plots and saves the plot in the log test directory.
Reads is a dictionary with keys as name of test and values as vector of cases.

column is the name as symbol of the column to plot

line_function is function used to determine the exact line point in a given place:
    :all - plot all the points
    :mean - plot mean of all the points
    :median - plot median of all the points
    :max - plot maximum of all the points
    :min - plot minimum of all the points
All of them except for :all will plot the std as well
"""
function plot_all(
    reads::Dict{String, Vector{DataFrames.DataFrame}},
    x_label::Symbol,
    y_label::Symbol,
    line_function::Symbol
)
    mkpath(ANALYSIS_DIR)
    infix_test = "[" * join([isa(infix_el, Tuple) ? join(infix_el, " or ") : infix_el for infix_el in TEST_INFIX_LIST], " & ") * "]"
    groups_text = "[" * join(GROUPS, " & ") * "]"
    plot_name = "function--$line_function  x--$x_label  y--$y_label  Prefix--$TEST_PREFIX"
    if infix_test != "[]"  # if there is infix
        plot_name *= "  Infix--$infix_test"
    end
    if groups_text != "[]"  # if there are groups
        plot_name *= "  Groups--$groups_text-$GROUPS_IN_LEGEND"
    end
    println("Plotting: $plot_name")
    plot_save_name = replace(plot_name, " " => "_")

    p = Plots.plot(
        legend=:topleft,
        xlabel=String(x_label),
        ylabel=String(y_label),
        title=plot_name,
        size=PLOT_SIZE,
        margin=PLOT_MARGIN,
        top_margin=PLOT_TOP_MARGIN,
    )
    current_colour = 1

    names = sort(collect(keys(reads)))
    groups = Dict{String, Vector{String}}()
    for name in names
        add_to_group!(groups, name, GROUPS)
    end

    was_there_previous_group = false
    sorted_group_names = sort(collect(keys(groups)))
    for group_name in sorted_group_names
        is_in_group = group_name == "" ? false : true

        if GROUPS_IN_LEGEND == :all || !is_in_group  # if group name is empty, it means that it is not in any group
            one_colour = false
            one_entry = false
        elseif GROUPS_IN_LEGEND == :col1
            one_colour = true
            one_entry = false
        elseif GROUPS_IN_LEGEND == :col1_ent1
            one_colour = true
            one_entry = true
        else
            throw("Unknown GROUPS_IN_LEGEND")
        end

        if was_there_previous_group
            plot_label_only!(p, "-------------------------------------------", DISTINT_COLOURS[current_colour], false, false)
        end

        # plot group_name
        if is_in_group
            plot_label_only!(p, group_name, DISTINT_COLOURS[current_colour], false, one_colour)
            was_there_previous_group = true
        end

        for name in groups[group_name]
            if !one_colour 
                current_colour = current_colour % length(DISTINT_COLOURS) + 1
            end
            cases = reads[name]
            all_cases = [(df[!, x_label], df[!, y_label]) for df in cases]
            common_x, y_values = get_common_x_and_vals(all_cases)  # common_x is vector, y_values is matrix (columnwise are timesteps, rowwise are cases)
            
            std_each_step = Statistics.std.(eachcol(y_values))

            if line_function == :all
                for row in eachrow(y_values)
                    plot_line_ribbon_only!(p, common_x, row, DISTINT_COLOURS[current_colour])
                end
            else
                if line_function == :mean
                    values = Statistics.mean.(eachcol(y_values))
                    ribbon = std_each_step
                elseif line_function == :median
                    values = Statistics.median.(eachcol(y_values))
                    ribbon = std_each_step
                elseif line_function == :max
                    values = maximum.(eachcol(y_values))
                    ribbon = (std_each_step, zeros(length(std_each_step))) # currently I plot just +std, idk if I should do +2*std
                elseif line_function == :min
                    ribbon = (zeros(length(std_each_step)), std_each_step) # currently I plot just +std, idk if I should do +2*std
                    values = minimum.(eachcol(y_values))
                else
                    throw("Unknown line function")
                end
                plot_line_ribbon_only!(p, common_x, values, DISTINT_COLOURS[current_colour], ribbon)
            end

            if !one_entry
                plot_label_only!(p, name, DISTINT_COLOURS[current_colour], is_in_group, !one_colour)
            end
        end

        if one_colour 
            current_colour = current_colour % length(DISTINT_COLOURS) + 1
        end
    end

    Plots.savefig(joinpath(ANALYSIS_DIR, "$plot_save_name.png"))
end

function plot_label_only!(p, label::String, colour, is_in_group::Bool, opacity=true)
    label_changed = is_in_group ? "|--" * label : label
    Plots.plot!(p, [], [], label=label_changed, color=colour, linewidth=LINE_WIDTH, opacity=(opacity ? 1.0 : 0.0))
end

function plot_line_ribbon_only!(p, x, y, colour, ribbon=[])
    if SHOW_STD && length(ribbon) > 0
        Plots.plot!(p, x, y, label="", ribbon=ribbon, color=colour, fillalpha=RIBBON_FILL_ALPHA, linewidth=LINE_WIDTH)
    else
        Plots.plot!(p, x, y, label="", color=colour, linewidth=LINE_WIDTH)
    end
end

"""
It will construct good group name and add the name to the group.
If there isnt aproppriate group, it will add it to the group with empty name ("").
"""
function add_to_group!(groups::Dict, name::String, groups_names::Vector{String})
    group_name = ""
    for group in groups_names
        if contains(name, group)
            # find text between group and text with _A so _ and big letter
            # get tmp group as things between group and end of string
            # do the job:
            tmp_group = find_group(name, group)
            group_name_tmp = group[end] == "=" ? group[1:end-1] : group
            
            if group_name != ""
                group_name *= " "
            end
            group_name *= group_name_tmp * "=" * tmp_group
        end
    end

    group_vector = get!(groups, group_name, Vector{String}())
    push!(group_vector, name)
end


"""
It extracts value of a given group from the text.
"""
function find_group(text, group_name)
    text = split(text, group_name)[2]
    if text[1] == '='
        text = text[2:end]
    end

    first_closing_bracket = findfirst(')', text)

    find_first_equal = findfirst('=', text)
    if find_first_equal === nothing
        return text[1:(first_closing_bracket-1)]
    end
    text = text[1:(find_first_equal - 1)]

    # find last _ before first equal
    find_last_ = findlast('_', text)
    if find_last_ === nothing
        throw("There is no _ before = in $text")
    end
    return text[1:(find_last_ - 1)]
end

"""
Get common x values and values for all cases.
It will work e.g. when we plot by evaluation number with different evaluation numbers for each case.
x_cases in all_cases are vectors of x values, y_cases are vectors of y values.
x_cases should be sorted
"""
 function get_common_x_and_vals(all_cases)
    # create set of all values in xes
    x_cases = [x for (x, _) in all_cases]
    y_cases = [y for (_, y) in all_cases]
    xes = Set{eltype(all_cases[1][1])}()
    for x_case in x_cases
        push!(xes, x_case...)
    end

    sorted_x = sort(collect(xes))
    vals = Matrix{Float32}(undef, length(all_cases), length(sorted_x))
    previous_ind = [1 for _ in 1:length(all_cases)]

    for step_id in axes(vals, 2)
        # actualize previous ind
        current_x = sorted_x[step_id]
        for (case_id, x_case) in enumerate(x_cases)
            while previous_ind[case_id] < length(x_case) && x_case[previous_ind[case_id] + 1] <= current_x
                previous_ind[case_id] += 1
            end
        end

        # fill vals
        for case_ind in eachindex(all_cases)
            vals[case_ind, step_id] = y_cases[case_ind][previous_ind[case_ind]]
        end        
    end

    return sorted_x, vals
end

function read_tests() :: Dict{String, Vector{DataFrames.DataFrame}}
    reads = Dict{String, Vector{DataFrames.DataFrame}}()
    for file in readdir(RESULTS_DIR)
        contains_satisfied = all(
            isa(infix, Tuple)
            ?
            any(conforms_to(file, infix_elem) for infix_elem in infix)
            :
            conforms_to(file, infix)
            for infix in TEST_INFIX_LIST
        )
        if endswith(file, TEST_POSTFIX) && startswith(file, TEST_PREFIX) && contains_satisfied
            file_whole = joinpath(RESULTS_DIR, file)
            df = CSV.read(file_whole, DataFrames.DataFrame)
            list = get!(reads, get_name_string(file), Vector{DataFrames.DataFrame}())
            push!(list, df)
        end
    end
    return reads
end

"""
Like contains but if first character is !, it negates the result.
"""
function conforms_to(file_name::String, what_to_confort::String) :: Bool
    if startswith(what_to_confort, "!")  # negation
        return !contains(file_name, what_to_confort[2:end])
    else
        return contains(file_name, what_to_confort)
    end
end

function run()
    reads = read_tests()
    plot_all(reads, X_LABEL, Y_LABEL, LINE_METHOD)
end

# ------------------------------------------------------------------------------------------------
# run
run()
