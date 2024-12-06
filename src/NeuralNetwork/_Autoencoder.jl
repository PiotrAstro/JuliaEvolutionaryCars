export Autoencoder

struct Autoencoder <: AbstractNeuralNetwork
    encoder::AbstractNeuralNetwork
    decoder::AbstractNeuralNetwork
    chained::Flux.Chain
    mmd_weight::AbstractFloat
    learning_rate::AbstractFloat
end

function Autoencoder(encoder::AbstractNeuralNetwork, decoder::AbstractNeuralNetwork; mmd_weight::AbstractFloat, learning_rate::AbstractFloat) :: Autoencoder
    return Autoencoder(encoder, decoder, Flux.Chain(get_Flux_representation(encoder), get_Flux_representation(decoder)), mmd_weight, learning_rate)
end

function get_parameters(nn::Autoencoder) :: Flux.Params
    return Flux.params(get_Flux_representation(nn.encoder), get_Flux_representation(nn.decoder))
end

function set_parameters!(nn::Autoencoder, parameters::Flux.Params)
    Flux.loadparams!(nn.encoder, parameters[1])
    Flux.loadparams!(nn.decoder, parameters[2])
end

function get_loss(nn::Autoencoder) :: Function
    return get_loss(nn.decoder)
end

function get_Flux_representation(nn::Autoencoder)
    return nn.chained
end

function predict(nn::Autoencoder, X::Array{Float32}) :: Array{Float32}
    # return Flux.testmode!(nn.layers(X))
    return nn.chained(X)
end

function copy(nn::Autoencoder) :: Autoencoder
    return Autoencoder(copy(nn.encoder), copy(nn.decoder); mmd_weight=nn.mmd_weight, learning_rate=nn.learning_rate)
end

function learn!(
    nn::Autoencoder,
    X::Array{Float32},
    Y::Array{Float32};
    epochs::Int = 10,
    batch_size::Int = 256,
    verbose::Bool = true
)
    nn_params = get_parameters(nn)
    nn_loss = get_loss(nn)
    encoder = get_Flux_representation(nn.encoder)
    decoder = get_Flux_representation(nn.decoder)

    optimiser = Flux.AdamW(nn.learning_rate)

    # shuffle x and y the same way
    perm = Random.randperm(size(X, 2))
    X = X[:, perm]
    Y = Y[:, perm]

    batches = [
        (
        X[:, i: (i+batch_size-1 <= size(X, 2) ? i+batch_size-1 : end)],
        Y[:, i: (i+batch_size-1 <= size(Y, 2) ? i+batch_size-1 : end)]
        ) for i in 1:batch_size:size(X, 2)
    ]

    for epoch in 1:epochs
        for (x, y) in batches
            gs = Flux.gradient(() -> custom_loss(encoder, decoder, nn_loss, nn.mmd_weight, x, y), nn_params)
            Flux.Optimise.update!(optimiser, nn_params, gs)
        end

        # print loss
        if verbose
            loss_combined_value = Statistics.mean(custom_loss(encoder, decoder, nn_loss, nn.mmd_weight, X, Y))
            loss_reconstruction_value = Statistics.mean(nn_loss(predict(nn, X), Y))
            println("Epoch: $epoch, Loss combined: $loss_combined_value, Loss reconstruction: $loss_reconstruction_value")
        end
    end
end

# Loss Function with Regularization
function custom_loss(encoder, decoder, base_loss, mmd_weight, x, y)
    z = encoder(x)
    x_hat = decoder(z)
    # Reconstruction loss
    recon_loss = base_loss(x_hat, y)
    # MMD regularization
    z_prior = randn(Float32, size(z))

    if mmd_weight == 0.0
        return recon_loss
    else
        mmd_loss = mmd_div(z, z_prior)
        # Total loss
        return recon_loss + mmd_weight * mmd_loss
    end
end

function mmd_div(
    x::AbstractArray,
    y::AbstractArray;
    kernel::Function=gaussian_kernel,
    kernel_kwargs::NamedTuple=NamedTuple()
)
    # Compute and return MMD divergence
    return StatsBase.mean(kernel(x, x; kernel_kwargs...)) +
           StatsBase.mean(kernel(y, y; kernel_kwargs...)) -
           2 * StatsBase.mean(kernel(x, y; kernel_kwargs...))
end # function

function gaussian_kernel(
    x::AbstractArray,
    y::AbstractArray;
    ρ=1.0f0,
    dims::Int=2
)
    # return Gaussian kernel
    return exp.(
        -Distances.pairwise(
            Distances.SqEuclidean(), x, y; dims=dims
        ) ./ ρ^2 ./ size(x, 1)
    )
end # function

# function compute_mmd(z, z_prior)
#     # Implement MMD computation
#     # For example, using a Gaussian kernel
#     batch_size = size(z, 2)
#     dimensions = Float32(size(z, 1))
#     xx = sum(abs2, z; dims=1)
#     yy = sum(abs2, z_prior; dims=1)
#     xy = z' * z_prior
#     # Compute distances
#     dxx = xx .+ xx' .- 2 * (z' * z)
#     dyy = yy .+ yy' .- 2 * (z_prior' * z_prior)
#     dxy = xx .+ yy' .- 2 * xy
#     # Gaussian kernel with bandwidth σ^2
#     sigma2 = 1.0
#     kxx = exp.(-dxx / (dimensions))
#     kyy = exp.(-dyy / (dimensions))
#     kxy = exp.(-dxy / (dimensions))
#     mmd = Flux.mean(kxx) + Flux.mean(kyy) - 2 * Flux.mean(kxy)
#     println("mmd: $mmd  kxx: $(Flux.mean(kxx))  kyy: $(Flux.mean(kyy))  kxy: $(Flux.mean(kxy))")
#     return mmd
# end
