
import Plots
using Plots.PlotMeasures
import DataFrames
import CSV
import Statistics


# ------------------------------------------------------------------------------------------------
# My params


TEST_DIR = joinpath("log", "parameters_tests_2025-01-04_19-26-01")
RESULTS_DIR = joinpath(TEST_DIR, "results")
ANALYSIS_DIR = joinpath(TEST_DIR, "analysis")

TEST_POSTFIX = ".csv"  # will be removed from plot entries
TEST_PREFIX = "logs_opt=StaGroGA_"  # will be removed from plot entries

COLUMN = :best_fitness
LINE_METHOD = :mean

# will stay in the plot entries, used for filtering
# entry should have all of these (there is and between them)
# but you can also use tuple inside, then there is or, e.g.
# TEST_INFIX_LIST = ["40", ("30", "50")]  ->  contains("40") && (contains("30") || contains("50"))
# usually you should use it like this TEST_INFIX_LIST = ["(MmdWei=0.0)"] 
TEST_INFIX_LIST = [("HclDis=complete", "HclDis=ward")]




# ------------------------------------------------------------------------------------------------
# Utils
DISTINT_COLOURS = colors = [
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
X_LABEL = "Generation"
Y_LABEL = "Fitness"

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
function plot(
    reads::Dict{String, Vector{DataFrames.DataFrame}},
    column::Symbol,
    line_function::Symbol
)
    mkpath(ANALYSIS_DIR)
    infix_test = "[" * join([isa(infix_el, Tuple) ? join(infix_el, " or ") : infix_el for infix_el in TEST_INFIX_LIST], " & ") * "]"
    plot_name = "function--$line_function  Col--$column  Prefix--$TEST_PREFIX  Infix--$infix_test"
    println("Plotting: $plot_name")
    plot_save_name = replace("_"*plot_name, " " => "_")

    p = Plots.plot(
        legend=:topleft,
        xlabel=X_LABEL,
        ylabel=Y_LABEL,
        title=plot_name,
        size=PLOT_SIZE,
        margin=PLOT_MARGIN,
        top_margin=PLOT_TOP_MARGIN,
    )
    current_colour = 1

    names = sort(collect(keys(reads)))
    for name in names
        list = reads[name]
        all_lists = [df[!, column] for df in list]
        max_length = maximum(length, all_lists)
        values_each_step = [[values[i] for values in all_lists if i <= length(values)] for i in 1:max_length]
        std_each_step = [Statistics.std(values) for values in values_each_step]

        if line_function == :all
            label_name = name
            for values in all_lists
                Plots.plot!(p, values, label=label_name, color=DISTINT_COLOURS[current_colour], linewidth=LINE_WIDTH)
                label_name = ""
            end
        elseif line_function == :mean
            Plots.plot!(p, Statistics.mean.(values_each_step), ribbon=std_each_step, label=name, color=DISTINT_COLOURS[current_colour], fillalpha=RIBBON_FILL_ALPHA, linewidth=LINE_WIDTH)
        elseif line_function == :median
            Plots.plot!(p, Statistics.median.(values_each_step), ribbon=std_each_step, label=name, color=DISTINT_COLOURS[current_colour], fillalpha=RIBBON_FILL_ALPHA, linewidth=LINE_WIDTH)
        elseif line_function == :max
            # ribbon = (2 .* std_each_point, zeros(length(std_each_point)))
            ribbon = (std_each_step, zeros(length(std_each_step))) # currently I plot just +std, idk if I should do +2*std
            Plots.plot!(p, maximum.(values_each_step), ribbon=ribbon, label=name, color=DISTINT_COLOURS[current_colour], fillalpha=RIBBON_FILL_ALPHA, linewidth=LINE_WIDTH)
        elseif line_function == :min
            # ribbon = (zeros(length(std_each_point)), 2 .* std_each_point)
            ribbon = (zeros(length(std_each_step)), std_each_step) # currently I plot just +std, idk if I should do +2*std
            Plots.plot!(p, minimum.(values_each_step), ribbon=ribbon, label=name, color=DISTINT_COLOURS[current_colour], fillalpha=RIBBON_FILL_ALPHA, linewidth=LINE_WIDTH)
        else
            throw("Unknown line function")
        end
        current_colour = current_colour % length(DISTINT_COLOURS) + 1
    end

    Plots.savefig(joinpath(ANALYSIS_DIR, "$plot_save_name.png"))
end

function read_tests() :: Dict{String, Vector{DataFrames.DataFrame}}
    reads = Dict{String, Vector{DataFrames.DataFrame}}()
    for file in readdir(RESULTS_DIR)
        contains_satisfied = all(
            isa(infix, Tuple) ? any(contains(file, infix_elem) for infix_elem in infix) :
            contains(file, infix)
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

function run()
    reads = read_tests()
    plot(reads, COLUMN, LINE_METHOD)
end

# ------------------------------------------------------------------------------------------------
# run
run()
