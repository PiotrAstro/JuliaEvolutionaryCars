module LuxVsFluxBenchmark

using MKL
import BenchmarkTools
import Flux
import Lux
import Random
import Statistics

export test

println("Number of Julia threads: $(Threads.nthreads())")

function create_flux_mlp()
    return Flux.Chain(
        Flux.Dense(64, 64, Flux.relu),
        Flux.Dense(64, 64, Flux.relu),
        Flux.Dense(64, 10)
    )
end

function create_lux_mlp()
    return Lux.Chain(
        Lux.Dense(64, 64, Lux.relu),
        Lux.Dense(64, 64, Lux.relu),
        Lux.Dense(64, 10)
    )
end

function benchmark_flux_forward(model, x, batch_size)
    println("\nFlux MLP - Batch size: $batch_size")
    ben = BenchmarkTools.@benchmark $model($x)
    display(ben)
end

function benchmark_lux_forward(model, x, batch_size, ps, st)
    println("\nLux MLP - Batch size: $batch_size")
    ben = BenchmarkTools.@benchmark Lux.apply($model, $x, $ps, $st)
    display(ben)
end

function benchmark_flux_backward(model, x, y, batch_size)
    println("\nFlux MLP Backward - Batch size: $batch_size")
    loss(x, y) = Flux.mse(model(x), y)
    ben = BenchmarkTools.@benchmark Zygote.gradient($loss, $x, $y)
    display(ben)
end

function benchmark_lux_backward(model, x, y, batch_size, ps, st)
    println("\nLux MLP Backward - Batch size: $batch_size")
    loss(x, y, ps, st) = sum(abs2, Lux.apply(model, x, ps, st)[1] .- y)
    ben = BenchmarkTools.@benchmark Zygote.gradient($loss, $x, $y, $ps, $st)
    display(ben)
end

function test()
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Create models
    flux_model = create_flux_mlp()
    lux_model = create_lux_mlp()
    
    # Initialize Lux model parameters and state
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, lux_model)
    
    # Test different batch sizes
    batch_sizes = [1, 32, 64, 128, 256]
    
    println("\n\n\n\n\n\n\n MLP Performance Tests")
    println("=====================================")
    
    for batch_size in batch_sizes
        println("\n\nTesting with batch size: $batch_size")
        
        # Generate random input and target
        x = Random.rand(Float32, 64, batch_size)
        y = Random.rand(Float32, 10, batch_size)
        
        # Benchmark forward pass
        benchmark_flux_forward(flux_model, x, batch_size)
        benchmark_lux_forward(lux_model, x, batch_size, ps, st)
        
        # Benchmark backward pass
        # benchmark_flux_backward(flux_model, x, y, batch_size)
        # benchmark_lux_backward(lux_model, x, y, batch_size, ps, st)
    end
end

end # module LuxVsFluxBenchmark

import .LuxVsFluxBenchmark
LuxVsFluxBenchmark.test()
