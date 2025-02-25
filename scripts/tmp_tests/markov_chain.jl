using MKL

module MarkovTests

using LinearAlgebra
using BenchmarkTools
import LoopVectorization
import Test

function example1()
    # Example matrix where we know λ=1 is an eigenvalue
    size_n = 1000
    matrix = rand(Float64, size_n, size_n)
    for vec_ in eachcol(matrix)
        # vec_ .-= minimum(vec_)
        vec_ ./= sum(vec_)
    end

    # Find nullspace of (A - I)
    println("\n\n\n\n\n\nTest1")
    println("Matrix:")
    display(matrix)
    result = nullspace(matrix - I)
    println("Nullspace:")
    display(result)
    println("nullspace * Matrix:")
    display(matrix * result)
end

function time()
    size_n = 1000
    matrix = rand(Float64, size_n, size_n)
    for vec_ in eachcol(matrix)
        # vec_ .-= minimum(vec_)
        vec_ ./= sum(vec_)
    end

    # Find nullspace of (A - I)
    println("\n\n\n\n\n\nTest time")
    result = nullspace(matrix - I)
    w_matrix = repeat(result', size_n, 1)
    display(w_matrix)

    matrix_trans = collect(transpose(matrix))
    
    inv_matrix = inv(I - matrix_trans + w_matrix)
    display(inv_matrix)

    b = @benchmark nullspace($matrix - I)
    display(b)

    b = @benchmark inv(I - $matrix_trans + $w_matrix)
    display(b)
end


function _markov_transition_matrix(membership_states_trajectories::Vector{Matrix{Float32}})::Matrix{Float32}
    states_n = size(membership_states_trajectories[1], 1)
    transition_matrix = zeros(Float32, states_n, states_n)

    for memberships in membership_states_trajectories
        LoopVectorization.@turbo for step in 1:(size(memberships, 2) - 1)
            for current_state in 1:states_n
                for next_state in 1:states_n
                    # here we use transposed matrix, to make it more cache friendly for Julia
                    transition_matrix[next_state, current_state] += memberships[current_state, step] * memberships[next_state, step + 1]
                end
            end
        end
    end

    for col in eachcol(transition_matrix)
        col ./= sum(col)
    end

    transposed_matrix = collect(transition_matrix')
    return transposed_matrix
end

function _markov_transition_matrix_newer(membership_states_trajectories::Vector{Matrix{Float32}})::Matrix{Float32}
    states_n = size(membership_states_trajectories[1], 1)
    transition_matrix = zeros(Float32, states_n, states_n)
    mul_cache = zeros(Float32, states_n, states_n)

    for memberships in membership_states_trajectories
        for step in 1:(size(memberships, 2) - 1)
            LinearAlgebra.mul!(mul_cache, view(memberships, :, step + 1), view(memberships, :, step)')
            transition_matrix .+= mul_cache
        end
    end

    for col in eachcol(transition_matrix)
        col ./= sum(col)
    end

    transposed_matrix = collect(transition_matrix')
    return transposed_matrix
end

function time_transition()
    membership_states_trajectories = [rand(Float32, 1000, 2000) for _ in 1:5]
    for memb in membership_states_trajectories
        for vec_ in eachcol(memb)
            vec_ ./= sum(vec_)
        end
    end

    result1 = _markov_transition_matrix(membership_states_trajectories)
    result2 = _markov_transition_matrix_newer(membership_states_trajectories)
    Test.@test result1 ≈ result2

    display(BenchmarkTools.@benchmark _markov_transition_matrix($membership_states_trajectories))
    display(BenchmarkTools.@benchmark _markov_transition_matrix_newer($membership_states_trajectories))
end

end

import .MarkovTests
MarkovTests.time()
MarkovTests.time_transition()