module SequenceTest

import Test
import BenchmarkTools

struct CarSequence
    states::Matrix{Float32}
end

function CarSequence(states::Vector{Vector{Float32}})
    states_n = length(states)
    feature_n = length(states[1])
    states_matrix = Matrix{Float32}(undef, feature_n, states_n)
    for (states_col, state) in zip(eachcol(states_matrix), states)
        states_col .= state
    end
    return CarSequence(states_matrix)
end

function CarSequence(seqs::Vector{CarSequence}) :: CarSequence
    states_n = 0
    for seq in seqs
        states_n += size(seq.states, 2)
    end
    feature_n = size(seqs[1].states, 1)
    
    states_matrix = Matrix{Float32}(undef, feature_n, states_n)
    col = 1
    for seq in seqs
        for seq_col in eachcol(seq.states)
            states_matrix[:, col] .= seq_col
            col += 1
        end
    end
    return CarSequence(states_matrix)
end

function CarSequence2(states::Vector{Vector{Float32}})
    states_matrix = reduce(hcat, states)
    return CarSequence(states_matrix)
end

function CarSequence2(seqs::Vector{CarSequence}) :: CarSequence
    states = reduce(hcat, [seq.states for seq in seqs])
    return CarSequence(states)
end

function test()
    feature_n = 30
    rand_vecs_n = 1000
    rand_seqs_n = 10

    random_vecs = [rand(Float32, feature_n) for _ in 1:rand_vecs_n]
    random_seqs = [CarSequence(rand(Float32, feature_n, rand_vecs_n)) for _ in 1:rand_seqs_n]

    Test.@test CarSequence(random_vecs).states == CarSequence2(random_vecs).states
    Test.@test CarSequence(random_seqs).states == CarSequence2(random_seqs).states

    println("\n\nBenchmarking vectors:")
    display(BenchmarkTools.@benchmark CarSequence($random_vecs))
    display(BenchmarkTools.@benchmark CarSequence2($random_vecs))

    println("\n\nBenchmarking sequences:")
    display(BenchmarkTools.@benchmark CarSequence($random_seqs))
    display(BenchmarkTools.@benchmark CarSequence2($random_seqs))
end

end  # module SequenceTest

import .SequenceTest

SequenceTest.test()