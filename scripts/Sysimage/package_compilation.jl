# should add this to settings: "julia.useCustomSysimage": true
# then it will automatically use sysimage when starting repl
# I think I shouldnt use sysimage on remote, it might be problematic with versions
# I have to be careful - when I change package version, t wont be changed unless I delete or recompile sys image!
import PackageCompiler

PACKAGES = [
    # "MuJoCo",
    # "MKL",
    # "LinearAlgebra",
    # "Random",
    "Images",
    # "Lux",
    "DataStructures",
    "Statistics",
    "StatsBase",
    # "Zygote",
    "Clustering",
    "Distances",
    # "Plots",
    # "Test",
    # "TestItems",
    # "Printf",
    # "LoggingExtras",
    # "DataFrames",
    "CSV",
    # "Profile",
    # "PProf",
    "FileIO",
    # "Dates",
    "Optimisers",
    "SimpleChains",
    # "BenchmarkTools",
    # "InteractiveUtils",
    # "JLD",
    "LoopVectorization",
]
SYSIMAGE_PATH = "JuliaSysimage.dll"
PRECOMPILE_SCRIPT = joinpath(@__DIR__, "precompilation.jl")

PackageCompiler.create_sysimage(
    PACKAGES;
    sysimage_path=SYSIMAGE_PATH,
    precompile_execution_file=PRECOMPILE_SCRIPT
)