module MarkovTests

using LinearAlgebra
using BenchmarkTools

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
    size_n = 500
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

end

import .MarkovTests
MarkovTests.time()