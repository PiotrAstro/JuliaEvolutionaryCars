module PlottingInterpret 

import Plots
using Plots.PlotMeasures
import DataFrames
import CSV
import Statistics


# ------------------------------------------------------------------------------------------------
# My params


TEST_DIR = joinpath("log", "parameters_tests_2025-02-17_15-17-17")
RESULTS_DIR = joinpath(TEST_DIR, "results")
ANALYSIS_DIR = joinpath(TEST_DIR, "analysis")

TEST_POSTFIX = ".csv"  # will be removed from plot entries
TEST_PREFIX = "o=ConStaGroSimGA_"  # will be removed from plot entries

Y_LABEL = :best_fitness
X_LABEL = :total_evaluations
LINE_METHOD = :p90  # :all, :mean, :median, :max, :min, :p25, :p90 - p is for percentiles 
SHOW_STD = false  # whether to show std ribbon, doesnt matter for :all

# By default [], so no GROUPS
# could be e.g. ["NClu", "MmdWei"] it will create groups for each combination of these, if entry doesnt have any of these, it will be a group on its own
GROUPS = ["NorMod", "RanMatMod"]
GROUPS_IN_LEGEND = :col1  # :all - different colours in groups, :col1 - one colour in groups, :col1_ent1 - one colour in groups and one entry in legend

# will stay in the plot entries, used for filtering
# TEST_INFIX_LIST = ["40", ("30", "!50")]  ->  contains("40") && (contains("30") || !contains("50"))
# usually you should use it like this TEST_INFIX_LIST = ["(MmdWei=0.0)"] 
# if you add ! as the first string index, it means not this one, e.g. TEST_INFIX_LIST = ["!40", "!50"] -> !contains("40") && !contains("50")
TEST_INFIX_LIST = ["Pam", "Sca", "NClu=40", ("DSum", "Min0")]


# ------------------------------------------------------------------------------------------------
# Utils
DISTINT_COLOURS = [
    # Your original colors
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
    "#c5b0d5",  # Light Purple
    
    # New distinct additions
    "#006400",  # Dark Green
    "#800080",  # Deep Purple
    "#FFD700",  # Gold
    "#FF1493",  # Deep Pink
    "#483D8B",  # Dark Slate Blue
    "#8B0000",  # Dark Red
    "#008B8B",  # Dark Cyan
    "#DC143C",  # Crimson
    "#556B2F",  # Dark Olive Green
    "#DAA520"   # Goldenrod
]
LINES = [
    :solid          # _______________
    :dash           # --------------
    :dot            # ................
    :dashdot        # _._._._._._._
    :dashdotdot     # _.._.._.._..
]
PLOT_SIZE = (6000, 3000)
LEGEND_POSITION = :bottom  # left, right, topleft, topright, bottomleft, bottomright, inside, bottom, outertopleft, outertopright, outerbottomleft, outerbottomright
PLOT_MARGIN = (maximum(PLOT_SIZE) / 150)mm
FONT_SIZE = 15
LEGEND_FONT_SIZE = 9
LEGEND_LINE_LENGTH = LEGEND_FONT_SIZE
LINE_WIDTH = 3
RIBBON_FILL_ALPHA = 0.1
GROUP_DELIMITER = "-----------------------------------------------"
GROUP_SUB_ENTRY_PREFIX = "|-- "

function get_name_string(file_name::String) :: String
    remove_prefix = replace(file_name, TEST_PREFIX => "")
    remove_case_extension = split(remove_prefix, "__")[1]
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
    infix_test = "[" * join([isa(infix_el, Tuple) ? "("*join(infix_el, ";")*")" : infix_el for infix_el in TEST_INFIX_LIST], "&") * "]"
    groups_text = "[" * join(GROUPS, "&") * "]"
    plot_name = "fun=$line_function x=$x_label y=$y_label Pre=$TEST_PREFIX"
    if infix_test != "[]"  # if there is infix
        plot_name *= " Inf=$infix_test"
    end
    if groups_text != "[]"  # if there are groups
        plot_name *= " Gro=$groups_text-$GROUPS_IN_LEGEND"
    end
    plot_name *= " rib=" * (SHOW_STD ? "T" : "F")
    println("Plotting: $plot_name")
    plot_save_name = replace(plot_name, " " => "_")

    p = Plots.plot(
        legend=LEGEND_POSITION,  # left, right, topleft, topright, bottomleft, bottomright, inside, bottom
        xlabel=String(x_label),
        ylabel=String(y_label),
        title=plot_name,
        legendfontsize=LEGEND_FONT_SIZE,
        tickfontsize=FONT_SIZE,
        guidefontsize=FONT_SIZE,
        titlefontsize=FONT_SIZE,
        size=PLOT_SIZE,
        margin=PLOT_MARGIN,
    )
    lines_ribbons_plotting = Dict()
    current_colour = 1
    current_line = 1

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
            plot_label_only!(p, GROUP_DELIMITER, DISTINT_COLOURS[current_colour], LINES[current_line], false, false)
        end

        # plot group_name
        if is_in_group
            plot_label_only!(p, group_name, DISTINT_COLOURS[current_colour], LINES[current_line], false, one_entry)
            was_there_previous_group = true
        end

        sorted_names = sort(groups[group_name])
        for name in sorted_names
            cases = reads[name]
            all_cases = [(df[!, x_label], df[!, y_label]) for df in cases]
            common_x, y_values = get_common_x_and_vals(all_cases)  # common_x is vector, y_values is matrix (columnwise are timesteps, rowwise are cases)
            
            std_each_step = Statistics.std.(eachcol(y_values))

            if line_function == :all
                for row in eachrow(y_values)
                    add_line_ribbon_only!(lines_ribbons_plotting, common_x, row, DISTINT_COLOURS[current_colour], LINES[current_line])
                end
            else
                if line_function == :mean
                    values = Statistics.mean.(eachcol(y_values))
                    ribbon = std_each_step
                else
                    if line_function == :median
                        quantile = 0.5
                    elseif line_function == :max
                        quantile = 1.0
                    elseif line_function == :min
                        quantile = 0.0
                    elseif String(line_function)[1] == 'p'
                        percentile = parse(Int, String(line_function)[2:end])
                        if percentile < 0 || percentile > 100
                            throw("Percentile should be between 0 and 100, currently is $percentile")
                        end
                        quantile = percentile / 100
                    else
                        throw("Unknown line function")
                    end

                    values = Statistics.quantile.(eachcol(y_values), quantile)
                    ribbon = (quantile * std_each_step, (1 - quantile) * std_each_step)
                end
                add_line_ribbon_only!(lines_ribbons_plotting, common_x, values, DISTINT_COLOURS[current_colour], LINES[current_line], ribbon)
            end

            if !one_entry
                plot_label_only!(p, name, DISTINT_COLOURS[current_colour], LINES[current_line], is_in_group, true)
            end
            
            if one_entry
                current_line = 1
            elseif one_colour
                current_line = next_line_style(current_line)
            else
                current_colour, current_line = next_colour_and_line_style(current_colour, current_line)
            end
        end

        if one_colour 
            current_colour, current_line = next_colour_and_line_style(current_colour, current_line)
            current_line = 1
        end
    end
    plot_lines_ribbons_only!(p, lines_ribbons_plotting)
    Plots.savefig(joinpath(ANALYSIS_DIR, "$plot_save_name.png"))
end

"""
It will return next colour and next linestyle.
"""
function next_colour_and_line_style(current_colour, current_linestyle) :: Tuple{Int, Int}
    current_colour = current_colour + 1
    if current_colour > length(DISTINT_COLOURS)
        current_colour = 1
        current_linestyle = next_line_style(current_linestyle)
    end
    return current_colour, current_linestyle
end

"""
It will return next linestyle.
"""
function next_line_style(current_linestyle) :: Int
    return current_linestyle % length(LINES) + 1
end

function plot_label_only!(p, label::String, colour, linestyle, is_in_group::Bool, opacity=true)
    label_changed = is_in_group ? GROUP_SUB_ENTRY_PREFIX * label : label
    Plots.plot!(p, [], [], label=label_changed, color=colour, linestyle=linestyle, linewidth=LINE_WIDTH, opacity=(opacity ? 1.0 : 0.0))
end

"""
It will plot line with ribbon.
If ribbon is set to [], it will plot only line.
"""
function add_line_ribbon_only!(lines_ribbons_plotting::Dict, x, y, colour, linestyle, ribbon=[])
    plots_args_kwargs = get!(lines_ribbons_plotting, :plots, Vector{Dict}())
    push!(plots_args_kwargs, 
        Dict(
            :args => (x, y),
            :kwargs => Dict(
                :label => "",
                :color => colour,
                :linestyle => linestyle,
                :linewidth => LINE_WIDTH,
            )
        ))
    if SHOW_STD && length(ribbon) > 0
        ribbons_args_kwargs = get!(lines_ribbons_plotting, :ribbons, Vector{Dict}())
        push!(ribbons_args_kwargs, 
            Dict(
                :args => (x, y),
                :kwargs => Dict(
                    :ribbon => ribbon,
                    :label => "",
                    :color => colour,
                    :fillalpha => RIBBON_FILL_ALPHA,
                    :linewidth => 0,
                )
            ))
    end
end

"""
Finally plot lines and ribbons in the right order.
"""
function plot_lines_ribbons_only!(p, lines_ribbons_plotting::Dict)
    if haskey(lines_ribbons_plotting, :ribbons)
        for ribbon_args_kwargs in lines_ribbons_plotting[:ribbons]
            Plots.plot!(p, ribbon_args_kwargs[:args]...; ribbon_args_kwargs[:kwargs]...)
        end
    end

    if haskey(lines_ribbons_plotting, :plots)
        for plot_args_kwargs in lines_ribbons_plotting[:plots]
            Plots.plot!(p, plot_args_kwargs[:args]...; plot_args_kwargs[:kwargs]...)
        end
    else
        throw("There are no plots to plot")
    end
end

"""
It will construct good group name and add the name to the group.
If there isnt aproppriate group, it will add it to the group with empty name ("").
"""
function add_to_group!(groups::Dict, name::String, groups_names::Vector)
    group_name = ""
    for group in groups_names
        if contains(name, group)
            # find text between group and text with _A so _ and big letter
            # get tmp group as things between group and end of string
            # do the job:
            tmp_group = find_group(name, group)
            group_name_tmp = group[end] == '=' ? group[1:end-1] : group
            
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

end # module

# ------------------------------------------------------------------------------------------------
# run
import .PlottingInterpret
PlottingInterpret.run()
