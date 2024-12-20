using Distances
import LinearAlgebra
using LinearAlgebra
using LoopVectorization
using Tullio
BLAS.set_num_threads(1)

function cosine_theirs(states)
    Distances.pairwise(Distances.CosineDist(), states, states)
end

function cosine_mine(states1, states2)
    # states = states .* inv.(sqrt.(sum(abs2, states; dims=1)))
    # states2 = states2 .* inv.(sqrt.(sum(abs2, states2; dims=1)))
    # return 1.0f0 .- states2' * states

    # tullio implementation
    @tullio dot[i,j] := states1[k,i] * states2[k,j]

    # Step 2: Compute the norms of each column for states1 and states2.
    # We sum the squares and then take the square root.
    @tullio norms1[i] := (states1[k,i])^2
    @tullio norms2[j] := (states2[k,j])^2
    norms1 = sqrt.(norms1)
    norms2 = sqrt.(norms2)

    # Step 3: Compute the cosine similarity and then the cosine distance.
    # Cosine similarity(i,j) = dot[i,j] / (norms1[i]*norms2[j])
    # Cosine distance(i,j) = 1 - Cosine similarity(i,j)
    @tullio out[i,j] := 1 - (dot[i,j] / (norms1[i]*norms2[j]))
    return out
end

function euclidean_theirs(states)
    Distances.pairwise(Distances.Euclidean(), states, states)
end

function euclidean_mine(states1, states2)
    # n1, n2 = size(states1, 2), size(states2, 2)
    # features = size(states1, 1)
    # result = Matrix{Float32}(undef, n1, n2)

    # @fastmath @inbounds for i in 1:n1
    #     for j in 1:n2
    #         s = 0.0f0
    #         @simd for k in 1:features
    #             # s += (states1[k, i] - states2[k, j]) ^ 2
    #             x = states1[k, i]
    #             y = states2[k, j]
    #             s += 2 * x * y - x * x - y * y
    #         end
    #         result[i, j] = s
    #     end
    # end
    # return result

    @tullio threads=false out[i,j] := (states1[k,i] - states2[k,j])^2 |> sqrt;
    return out
end

function city_block_mine(states1, states2)
    # result = Matrix{Float32}(undef, size(states1, 2), size(states2, 2))

    # @inbounds for i in 1:size(states1, 2)
    #     for j in 1:size(states2, 2)
    #         s = 0.0f0
    #         @simd for k in 1:size(states1, 1)
    #             s += abs(states1[k, i] - states2[k, j])
    #         end
    #         s = (s)
    #         result[i, j] = s
    #     end
    # end

    # return result
    @tullio threads=false out[i,j] := abs(states1[k,i] - states2[k,j]);
    return out
end

function vec_eucl(state1, state2)
    return sqrt(sum(abs2, state1 - state2))
end

function vec_eucl_tullio(state1, state2)
    s = 0.0f0
    @inbounds @simd for i in 1:length(state1)
        s += (state1[i] - state2[i])^2
    end
    return sqrt(s)
end

function tests()
    random_states = rand(Float32, 30, 10)
    random_states2 = rand(Float32, 30, 1000)
    n = 1_00
    # x = 0.0f0


    # view_vector1 = @view random_states[:, 1]
    # view_vector2 = @view random_states[:, 2]

    # @time for _ in 1:(n*100)
    #     x = vec_eucl(view_vector1, view_vector2)
    # end

    # @time for _ in 1:(n*100)
    #     x = vec_eucl_tullio(view_vector1, view_vector2)
    # end

    # @time for _ in 1:(n*100)
    #     x = Distances.Euclidean()(view_vector1, view_vector2)
    # end

    println("cosine")
    @time for _ in 1:n
        t = Distances.pairwise(Distances.CosineDist(), random_states, random_states2)
    end

    println("cosine_mine")
    @time for _ in 1:n
        t = cosine_mine(random_states, random_states2)
    end

    println("euclidean")
    @time for _ in 1:n
        t = Distances.pairwise(Distances.Euclidean(), random_states, random_states2)
    end
    
    println("euclidean_sq")
    @time for _ in 1:n
        t = Distances.pairwise(Distances.SqEuclidean(), random_states, random_states2)
    end

    println("euclidean_mine")
    @time for _ in 1:n
        t = euclidean_mine(random_states, random_states2)
    end

    println("city_block")
    @time for _ in 1:n
        t = Distances.pairwise(Distances.Cityblock(), random_states, random_states2)
    end

    println("city_block_mine")
    @time for _ in 1:n
        t = city_block_mine(random_states, random_states2)
    end
end

tests()