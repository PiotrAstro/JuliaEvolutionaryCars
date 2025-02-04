module BitArrayTest

function access_in_loop(matrix, what_to_access)
    result = false
    for (i, j) in what_to_access
        @inbounds result = result || matrix[i, j]
    end
    return result
end

using BenchmarkTools
function run_one_test()
    y_dims = 600
    x_dims = 800
    number_of_what_to_search = 100
    random_bool_matrix = rand(Bool, y_dims, x_dims)
    random_bitmatrix = BitArray(random_bool_matrix)
    what_to_access = Vector{Tuple{Int, Int}}()
    for i in rand(1:y_dims, number_of_what_to_search)
        push!(what_to_access, (i, rand(1:x_dims)))
    end
    println(Base.summarysize(random_bool_matrix))
    println(Base.summarysize(random_bitmatrix))


    b = @benchmark access_in_loop($random_bool_matrix, $what_to_access)
    display(b)

    b = @benchmark access_in_loop($random_bitmatrix, $what_to_access)
    display(b)
end

end # module

import .BitArrayTest
BitArrayTest.run_one_test()