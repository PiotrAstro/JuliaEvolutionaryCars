module MulAndMembership

using BenchmarkTools
import Distances
using LoopVectorization

function test_membership()
    rand_values = (rand(10, 10) .- 0.5) .* 2

    rand_normalized = copy(rand_values)
    for col in eachcol(rand_normalized)
        sum = 0.0
        for v in col
            sum += v ^ 2
        end
        col ./= sum
    end
    cosine_values = rand_normalized' * rand_normalized
    mul_values = rand_values' * rand_values

    softmax_result = copy(cosine_values)
    softmax!(softmax_result)
    softmax_mul = copy(mul_values)
    softmax!(softmax_mul)

    mval = 2
    mval_result = copy(cosine_values)
    for col in eachcol(mval_result)
        for val_ind in eachindex(col)
            dist = 1.0 - col[val_ind] + 1e-6
            col[val_ind] = (1.0 / dist) ^ mval
        end
        col ./= sum(col)
    end
    
    println("\n\n\n\n")

    println("\n\nrand_values")
    display(rand_values)

    println("\n\nmul")
    display(mul_values)

    println("\n\nsoftmax_mul")
    display(softmax_mul)

    println("\n\ncosine_values")
    display(cosine_values)

    println("\n\nsoftmax_result")
    display(softmax_result)

    println("\n\nmval_result")
    display(mval_result)
end

function softmax!(x::Matrix)
    for col in eachcol(x)
        col .= exp.(col)
        col ./= sum(col)
    end
end

function max_speed_test()
    vector = rand(Float32, 1000)

    println("\n\n\nStarting max speed test")
    println("Vector size: $(length(vector))")

    println("Library function")
    display(@benchmark maximum($vector))
    println("Custom function")
    display(@benchmark custom_maximum($vector))
end

function custom_maximum(vector::Vector{Float32})
    max_val = typemin(Float32)
    # for val in vector
    #     max_val = ifelse(val > max_val, val, max_val)
    # end

    @inbounds for i in eachindex(vector)
        @fastmath max_val = ifelse(vector[i] > max_val, vector[i], max_val)
    end

    # @turbo for i in eachindex(vector)
    #     max_val = ifelse(vector[i] > max_val, vector[i], max_val)
    # end
    return max_val
end

function mval_test()
    vector = rand(Float32, 9)
    println("Vector size: $(length(vector))")
    println("LoopVectorization")
    display(@benchmark loop_vec_mval($vector, Val(2)))
    println("Custom")
    display(@benchmark normal_mval($vector, Val(2)))
end

function loop_vec_mval(vector::Vector{Float32}, ::Val{MVAL}) where {MVAL}
    # LoopVectorization.@turbo for i in eachindex(vector)
    #     distance = abs(1.0f0 - vector[i]) + EPSILON_EXEMPLARBASEDNN
    #     vector[i] = (1.0f0 / distance) ^ mval
    # end
    LoopVectorization.@turbo for i in eachindex(vector)
        distance = abs(1.0f0 - vector[i]) + Float32(1e-6)
        vector[i] = (1.0f0 / distance) ^ MVAL
    end
    vector .*= 1.0f0 / sum(vector)
end

function normal_mval(vector::Vector{Float32}, ::Val{MVAL}) where {MVAL}
    @fastmath @inbounds @simd for i in eachindex(vector)
        distance = abs(1.0f0 - vector[i]) + Float32(1e-6)
        vector[i] = (1.0f0 / distance) ^ MVAL
    end
    vector .*= 1.0f0 / sum(vector)
end


end

import .MulAndMembership
MulAndMembership.mval_test()
# MulAndMembership.test_membership()