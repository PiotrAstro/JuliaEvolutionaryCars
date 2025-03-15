module MulAndMembership

import Distances

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

end

import .MulAndMembership
MulAndMembership.test_membership()