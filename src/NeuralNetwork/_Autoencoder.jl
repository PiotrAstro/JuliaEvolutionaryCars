export Autoencoder

struct Autoencoder{E<:AbstractNeuralNetwork, D<:AbstractNeuralNetwork } <: AbstractNeuralNetwork
    encoder::E
    decoder::D
    chained::Flux.Chain
    mmd_weight::Float64
    learning_rate::Float64
end

function Autoencoder(
        encoder::E,
        decoder::D;
        mmd_weight::Float64,
        learning_rate::Float64
    ) where {E<:AbstractNeuralNetwork, D<:AbstractNeuralNetwork}
    return Autoencoder{E, D}(encoder, decoder, Flux.Chain(get_Flux_representation(encoder), get_Flux_representation(decoder)), mmd_weight, learning_rate)
end

function get_neural_network(name::Val{:Autoencoder})
    return Autoencoder
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

function copy(nn::Autoencoder)
    return Autoencoder(copy(nn.encoder), copy(nn.decoder); mmd_weight=nn.mmd_weight, learning_rate=nn.learning_rate)
end

function learn!(
    nn::Autoencoder,
    X::Array{Float32},
    Y::Array{Float32};
    epochs::Int = 10,
    batch_size::Int = 256,
    verbose::Bool = false
)
    nn_loss = get_loss(nn)
    encoder = get_Flux_representation(nn.encoder)
    decoder = get_Flux_representation(nn.decoder)
    chained = nn.chained

    # Set up the optimizer and optimizer state
    # (params, _) = Flux.setup(encoder, decoder)
    optimiser_encoder = Optimisers.AdamW(nn.learning_rate)
    optimiser_state = Flux.setup(optimiser_encoder, chained)

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
            # Compute gradients
            grads = Flux.gradient((ch) -> custom_loss(ch[1], ch[2], nn_loss, nn.mmd_weight, x, y), chained)
            # Update the model parameters and optimiser state
            Flux.update!(optimiser_state, chained, grads[1])
        end

        # print loss
        if verbose
            loss_combined_value = Statistics.mean(custom_loss(encoder, decoder, nn_loss, nn.mmd_weight, X, Y))
            loss_reconstruction_value = Statistics.mean(nn_loss(predict(nn, X), Y))
            Logging.@info "Epoch: $epoch, Loss combined: $loss_combined_value, Loss reconstruction: $loss_reconstruction_value\n"
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