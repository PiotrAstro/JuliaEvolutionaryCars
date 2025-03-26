using MKL
using LinearAlgebra
import BenchmarkTools

println("BLAS kernel: $(BLAS.get_config())")
println("Number of BLAS threads: $(BLAS.get_num_threads())")
println("Number of Julia threads: $(Threads.nthreads())")

function benchmark_matrix_mul(matrix1, matrix2, name)
    println("\n\nTesting $name")
    println("Matrix dimensions: $(size(matrix1)) × $(size(matrix2))")
    
    # Test regular multiplication
    println("\nRegular multiplication (A × B):")
    ben = BenchmarkTools.@benchmark $matrix1 * $matrix2
    display(ben)
    
    # Test with first matrix transposed
    println("\nFirst matrix transposed (A' × B):")
    ben = BenchmarkTools.@benchmark $(matrix1') * $matrix2
    display(ben)
    
    # Test with second matrix transposed
    println("\nSecond matrix transposed (A × B'):")
    ben = BenchmarkTools.@benchmark $matrix1 * $(matrix2')
    display(ben)
    
    # Test with both matrices transposed
    println("\nBoth matrices transposed (A' × B'):")
    ben = BenchmarkTools.@benchmark $(matrix1') * $(matrix2')
    display(ben)
end

function test_different_sizes()
    # Test with square matrices
    matrix1 = rand(1000, 1000)
    matrix2 = rand(1000, 1000)
    benchmark_matrix_mul(matrix1, matrix2, "Square matrices (1000×1000)")
    
    # Test with larger matrices
    matrix1 = rand(2000, 2000)
    matrix2 = rand(2000, 2000)
    benchmark_matrix_mul(matrix1, matrix2, "Large square matrices (2000×2000)")
end

function test_different_threads()
    # Use square matrices for thread testing to avoid dimension issues
    matrix1 = rand(1000, 1000)
    matrix2 = rand(1000, 1000)
    
    for blas_threads in [1, 2, 4, 8]
        println("\n\n\nTesting with $blas_threads BLAS threads")
        BLAS.set_num_threads(blas_threads)
        benchmark_matrix_mul(matrix1, matrix2, "Square matrices with $blas_threads BLAS threads")
    end
end

function main()
    println("\n\n\n\n\n\n\n Matrix Multiplication Performance Tests")
    println("=====================================")
    
    println("\nTesting different matrix sizes:")
    test_different_sizes()
    
    println("\n\nTesting different thread configurations:")
    test_different_threads()
end

main() 