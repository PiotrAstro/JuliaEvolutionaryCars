module LuxFluxSimpleChains

using MKL
import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

import Lux
# import Flux
import Random
import BenchmarkTools
using SimpleChains

function mlp_test()
    batch_size = 5
    size_n = 256
    layers_n = 2

    # flux_model = Flux.Chain(
    #     [Flux.Dense(size_n => size_n, relu) for _ in 1:layers_n]...,
    # )

    lux_model = Lux.Chain(
        [Lux.Dense(size_n => size_n, relu) for _ in 1:layers_n]...,
    )
    lx_ps, lx_st = Lux.setup(Random.Xoshiro(rand(Int)), lux_model)

    adaptor = Lux.ToSimpleChainsAdaptor((SimpleChains.static(size_n),), true)
    simple_model = adaptor(lux_model)
    lxsc_ps, lxsc_st = Lux.setup(Random.Xoshiro(rand(Int)), simple_model)

    random_input = randn(Float32, size_n, batch_size)

    println("\n\n\n\nMLP test: features: $(size_n), layers: $(layers_n) batch_size: $(batch_size)")

    # println("\nFlux model:")
    # display(BenchmarkTools.@benchmark ($flux_model)($random_input))

    println("\nLux model:")
    display(BenchmarkTools.@benchmark Lux.apply($lux_model, $random_input, $lx_ps, $lx_st))

    println("\nSimpleChains model:")
    display(BenchmarkTools.@benchmark Lux.apply($simple_model, $random_input, $lxsc_ps, $lxsc_st))
end

function count_params(params)
    total_params = 0
    for param in params
        if isa(param, Array)
            total_params += length(param)
        else
            total_params += count_params(param)
        end
    end
    return total_params
end

function conv_test()
    img_size=512
    channels=16
    kernel_size=3
    batch_size=1
    conv_layers=2
    # Calculate output dimensions after convolutions and pooling
    final_size = img_size
    for _ in 1:conv_layers
        final_size = div(final_size - kernel_size + 1, 2)  # Conv + maxpool halves dimension
    end
    final_feats = channels * final_size * final_size

    # Create random input
    random_input = randn(Float32, img_size, img_size, 1, batch_size)
    
    # Flux model
    # flux_model = Flux.Chain(
    #     [Flux.Chain(
    #         Flux.Conv((kernel_size, kernel_size), (i == 1 ? 1 : channels) => channels, relu),
    #         Flux.MaxPool((2, 2))
    #     ) for i in 1:conv_layers]...,
    #     Flux.flatten,
    #     Flux.Dense(final_feats => 10)
    # )
    
    # Lux model
    lux_model = Lux.Chain(
        [Lux.Chain(
            Lux.Conv((kernel_size, kernel_size), (i == 1 ? 1 : channels) => channels, relu),
            Lux.MaxPool((2, 2))
        ) for i in 1:conv_layers]...,
        Lux.FlattenLayer(3),
        Lux.Dense(final_feats => 10)
    )
    lx_ps, lx_st = Lux.setup(Random.default_rng(), lux_model)
    
    # SimpleChains model (using adaptor)
    # adaptor = Lux.ToSimpleChainsAdaptor((SimpleChains.static(img_size),SimpleChains.static(img_size),SimpleChains.static(1)))
    adaptor = Lux.ToSimpleChainsAdaptor((img_size,img_size,1), true)
    simple_model = adaptor(lux_model)
    lxsc_ps, lxsc_st = Lux.setup(Random.default_rng(), simple_model)
    # display(Lux.trainmode(lxsc_st))

    # Calculate parameters
    total_params = length(lxsc_ps.params)
    
    println("\n\n\n\nCNN test: img_size: $(img_size), channels: $(channels), " *
            "kernel: $(kernel_size), conv_layers: $(conv_layers), batch_size: $(batch_size)")
    println("Total parameters: ~$(round(total_params/1000, digits=1))K")
    
    println("\nSimpleChains model:")
    display(BenchmarkTools.@benchmark Lux.apply($simple_model, $random_input, $lxsc_ps, $lxsc_st))

    # println("\nFlux model:")
    # display(BenchmarkTools.@benchmark ($flux_model)($random_input))
    
    println("\nLux model:")
    display(BenchmarkTools.@benchmark Lux.apply($lux_model, $random_input, $lx_ps, $lx_st))
end

end

import .LuxFluxSimpleChains
LuxFluxSimpleChains.conv_test()
LuxFluxSimpleChains.mlp_test()