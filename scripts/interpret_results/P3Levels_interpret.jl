import Plots
import DataFrames
import CSV
import Statistics

LOGS_DIR = "log"
TEST_DIR = "parameters_tests_2024-12-11_10-41-13"
TEST_PREFIX = "logs_opt=StaGroGA_"

# ----------------------------------------------------------------------------------
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
It plots and displays the plot.
Reads is a dictionary with keys as name of test and values as vector of cases.

column is the name as symbol of the column to plot

line_function is function used to determine the exact line point in a given place, e.g. mean, median, max, min, etc. We will automatically plot std

It can be nothing, then we will plot all the points without std

"""
function plot(
    reads::Dict{String, Vector{DataFrames.DataFrame}},
    column::Symbol,
    line_function::Union{Function, Nothing}=nothing
) :: Plots.Plot
    p = Plots.plot()
    for (name, list) in reads
        if isnothing(line_function)
            for df in list
                Plots.plot!(p, df[!, column], label=name)
            end
        end
    end
    return p
end

reads = Dict{String, Vector{DataFrames.DataFrame}}()
for file in readdir(joinpath(LOGS_DIR, TEST_DIR))
    if occursin(".csv", file) && startswith(file, TEST_PREFIX)
        df = CSV.read(LOGS_DIR * file)
        list = get!(reads, get_name_string(file), Vector{DataFrames.DataFrame}())
        push!(list, df)
    end
end
