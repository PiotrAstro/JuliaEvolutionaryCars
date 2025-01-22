using BenchmarkTools
using LinearAlgebra
using LoopVectorization

function my_norm(x::Matrix{Float32})
    x .*= inv.(sqrt.(sum(abs2, x; dims=1)))
end

function my_norm_2(x::Matrix{Float32})
    for col_ind in axes(x, 2)
        sum_squared = 0.0f0
        @turbo for row_ind in axes(x, 1)
            sum_squared += x[row_ind, col_ind] ^ 2
        end
        sum_squared = 1.0f0 / sqrt(sum_squared)
        @turbo for row_ind in axes(x, 1)
            x[row_ind, col_ind] *= sum_squared
        end
    end
end

function normalize_their(x::Matrix{Float32})
    for col in eachcol(x)
        normalize!(col)
    end
end

function test()
    abc = rand(Float32, 32, 5)
    display(abc)
    
    abc_their = copy(abc)
    normalize_their(abc_their)
    display(abc_their)
    sleep(2)

    abc_mine = copy(abc)
    my_norm(abc_mine)
    display(abc_mine)
    sleep(2)

    abc_mine_2 = copy(abc)
    my_norm_2(abc_mine_2)
    display(abc_mine_2)
    sleep(2)

    b = @benchmark normalize_their($abc)
    display(b)

    b = @benchmark my_norm($abc)
    display(b)

    b = @benchmark my_norm_2($abc)
    display(b)
end

test()