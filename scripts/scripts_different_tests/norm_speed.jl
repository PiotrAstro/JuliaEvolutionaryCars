using BenchmarkTools
using LinearAlgebra

function my_norm(x::Matrix{Float32})::Matrix{Float32}
    x .*= inv.(sqrt.(sum(abs2, x; dims=1)))
end

function normalize_their(x::Matrix{Float32})
    for col in eachcol(x)
        normalize!(col)
    end
end

function test()
    abc = rand(Float32, 1000, 1000)
    display(abc)

    b = @benchmark normalize_their($abc)
    display(b)

    b = @benchmark my_norm($abc)
    display(b)
end

test()