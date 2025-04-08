# ------------------------------------------------------------------------------------------------
# state sequences managers

export AbstractStateSequence, get_length, copy_nth_state, get_sequence_with_ids, remove_nth_state, get_nn_input

# INTERNAL - internal concrete type, environment will receive it for reaction, e.g. Vector{Float32} or Array{Float32, 3} for rgb images
abstract type AbstractStateSequence{INTERNAL} end

# AbstractStateSequence should have the same type in return, it should take some iterable object
function (::AbstractStateSequence{INTERNAL})(states::AbstractVector{INTERNAL}) :: AbstractStateSequence{INTERNAL} where {INTERNAL}
    throw("unimplemented")
end

# AbstractStateSequence should have the same type in return
function (::AbstractStateSequence{INTERNAL})(seqs::AbstractVector{AbstractStateSequence{INTERNAL}}) :: AbstractStateSequence{INTERNAL} where {INTERNAL}
    throw("unimplemented")
end

function get_length(seq::ASSEQ) :: Int where {ASSEQ<:AbstractStateSequence}
    throw("unimplemented")
end

function copy_nth_state(seq::ASSEQ, n::Int) :: INTERNAL where {INTERNAL, ASSEQ<:AbstractStateSequence{INTERNAL}} 
    throw("unimplemented")
end

function get_sequence_with_ids(seq::ASSEQ, ids::AbstractVector{Int}) :: ASSEQ where {ASSEQ<:AbstractStateSequence}
    throw("unimplemented")
end

# It shouldnt change the original sequence
function remove_nth_state(seq::ASSEQ, n::Int) :: ASSEQ where {ASSEQ<:AbstractStateSequence}
    throw("unimplemented")
end

function get_nn_input(seq::ASSEQ) where {ASSEQ<:AbstractStateSequence}
    throw("unimplemented")
end

"""
Returns some generator
things in this generator are ready to put them into neural network
"""
function prepare_batches(seqs::ASSEQ, batch_size::Int; permutation::Union{Nothing, AbstractVector{Int}}=nothing) where {ASSEQ<:AbstractStateSequence}
    throw("unimplemented")
end

# ---------------------------------------------------------------------------------------
# MatrixASSEQ - for simple 1d data, it is matrix, cause we add batch dimension to it
struct MatrixASSEQ <: AbstractStateSequence{Vector{Float32}}
    states::Matrix{Float32}
end

function MatrixASSEQ(states::AbstractVector{Vector{Float32}}) :: MatrixASSEQ
    states_matrix = reduce(hcat, states)
    return MatrixASSEQ(states_matrix)
end

function MatrixASSEQ(seqs::Vector{MatrixASSEQ}) :: MatrixASSEQ
    states = reduce(hcat, [seq.states for seq in seqs])
    return MatrixASSEQ(states)
end

function copy_nth_state(seq::MatrixASSEQ, n::Int) :: Vector{Float32}
    return seq.states[:, n]
end

function get_length(seq::MatrixASSEQ) :: Int
    return size(seq.states, 2)
end

function get_sequence_with_ids(seq::MatrixASSEQ, ids::AbstractVector{Int}) :: MatrixASSEQ
    return MatrixASSEQ(seq.states[:, ids])
end

function remove_nth_state(seq::MatrixASSEQ, n::Int) :: MatrixASSEQ
    states = seq.states[:, 1:end .!= n]
    return MatrixASSEQ(states)
end

function get_nn_input(seq::MatrixASSEQ)
    return seq.states
end

function prepare_batches(seqs::MatrixASSEQ, batch_size::Int; permutation::Union{Nothing, AbstractVector{Int}}=nothing)
    X = seqs.states
    if !isnothing(permutation)
        X = X[:, perm]
    end
    columns_n = size(X, 2)

    # I should think about changing it to view
    batches = (
        X[ :, i: (i+batch_size-1 <= columns_n ? (i+batch_size-1) : columns_n)]
            for i in 1:batch_size:columns_n
    )
    return batches
end
