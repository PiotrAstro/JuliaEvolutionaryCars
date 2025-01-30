using MKL
using LinearAlgebra
import BenchmarkTools
println("BLAS kernel: $(BLAS.get_config())")
println(Threads.nthreads())

function blas_alone(matrix1, matrix2)
    for i in 1:4
        println("\n\nBLAS $i")
        BLAS.set_num_threads(i)
        ben = BenchmarkTools.@benchmark $matrix1 * $matrix2
        display(ben)
    end
end

function blas_julia(matrix1, matrix2)
    blases = [1, 4]
    jules = [1, 4]
    times = 16
    for jul in jules
        for blas in blases
            println("\n\nBLAS $blas, Julia $jul")
            BLAS.set_num_threads(blas)
            per_threads_n = div(times, jul)
            ben = BenchmarkTools.@benchmark calculate_n($matrix1, $matrix2, $jul, $per_threads_n)
            display(ben)
        end
    end
end

function calculate_n(matrix1, matrix2, threads_n, per_threads_n)
    Threads.@threads for i in 1:threads_n
        for j in 1:per_threads_n
            matrix1 * matrix2
        end
    end
end

function test()
    matrix1 = rand(1000, 1000)
    matrix2 = rand(1000, 1000)

    println("\n\n\n\n\n\n\n Tests")
    blas_julia(matrix1, matrix2)
    
    # blas_alone(matrix1, matrix2)
end

test()

