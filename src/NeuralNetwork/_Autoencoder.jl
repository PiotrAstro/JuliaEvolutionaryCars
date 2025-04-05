export Autoencoder

struct Autoencoder{E<:AbstractNeuralNetwork, D<:AbstractNeuralNetwork, C} <: AbstractNeuralNetwork
    encoder::E
    decoder::D
    internal::C
    mmd_weight::Float64
    learning_rate::Float64
    weight_decay::Float64
end


function Autoencoder(
        encoder::E,
        decoder::D;
        mmd_weight::Float64,
        learning_rate::Float64,
        weight_decay::Float64=0.0
    ) where {E<:AbstractNeuralNetwork, D<:AbstractNeuralNetwork}
    internal = Lux.Chain(
        encoder=get_lux_representation(encoder), 
        decoder=get_lux_representation(decoder)
    )

    return Autoencoder(
        encoder,
        decoder,
        internal,
        mmd_weight,
        learning_rate,
        weight_decay
    )
end

function get_neural_network(name::Val{:Autoencoder})
    return Autoencoder
end

function predict(nn::Autoencoder, X::AbstractStateSequence) :: Array{Float32}
    # return Flux.testmode!(nn.layers(X))
    internal = get_nn_input(X)
    encoded = predict(nn.encoder, internal)
    decoded = predict(nn.decoder, encoded)
    return decoded
end

function get_loss(nn::Autoencoder) :: Function
    return get_loss(nn.decoder)
end

function copy(nn::Autoencoder)
    return Autoencoder(
        copy(nn.encoder),
        copy(nn.decoder),
        nn.internal,
        nn.mmd_weight,
        nn.learning_rate,
        nn.weight_decay
    )
end

function learn!(
    nn::Autoencoder,
    X::ASSEQ;
    epochs::Int = 10,
    batch_size::Int = 256,
    verbose::Bool = false
) where {ASSEQ<:AbstractStateSequence}
    nn_loss = get_loss(nn)
    ps = (;
        encoder=copy_parameters(nn.encoder),
        decoder=copy_parameters(nn.decoder)
    )
    st = (;
        encoder=copy_state(nn.encoder),
        decoder=copy_state(nn.decoder)
    )
    st = Lux.trainmode(st)

    # Set up the optimizer and optimizer state
    optimiser_encoder = Optimisers.AdamW(;eta=nn.learning_rate, lambda=nn.weight_decay)
    train_state = Lux.Training.TrainState(nn.internal, ps, st, optimiser_encoder)
    ad = Lux.AutoZygote()
    loss_fn = (model, ps, st, x) -> custom_loss(model, ps, st, nn_loss, nn.mmd_weight, x)

    # shuffle x and y the same way
    all_inputs = nothing
    batches = prepare_batches(X, batch_size)

    for epoch in 1:epochs
        for x in batches
            (_, loss, _, train_state) = Lux.Training.single_train_step!(
                ad, loss_fn, x, train_state
            )
        end

        # print loss
        if verbose
            if isnothing(all_inputs)
                all_inputs = get_nn_input(X)
            end
            st_ = Lux.testmode(st)
            loss_combined_value, _ = custom_loss(nn.internal, ps, st_, nn_loss, nn.mmd_weight, all_inputs)
            predicted, _ = Lux.apply(nn.internal, all_inputs, ps, st_)
            loss_reconstruction_value = nn_loss(predicted, all_inputs)
            Logging.@info "Epoch: $epoch, Loss combined: $loss_combined_value, Loss reconstruction: $loss_reconstruction_value\n"
        end
    end

    st = Lux.testmode(train_state.states)
    set_parameters!(nn.encoder, train_state.parameters.encoder)
    set_parameters!(nn.decoder, train_state.parameters.decoder)
    set_state!(nn.encoder, st.encoder)
    set_state!(nn.decoder, st.decoder)
end

# Loss Function with Regularization
function custom_loss(model, ps, st, base_loss, mmd_weight, x)
    z, st_encoder = Lux.apply(model.encoder, x, ps.encoder, st.encoder)
    x_hat, st_decoder = Lux.apply(model.decoder, z, ps.decoder, st.decoder)
    st = (;
        encoder=st_encoder,
        decoder=st_decoder
    )
    # Reconstruction loss
    recon_loss = base_loss(x_hat, x)
    # MMD regularization
    z_prior = randn(Float32, size(z))

    if mmd_weight == 0.0
        return recon_loss, st, nothing
    else
        mmd_loss = mmd_div(z, z_prior)
        # Total loss
        return (recon_loss + mmd_weight * mmd_loss), st, nothing
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