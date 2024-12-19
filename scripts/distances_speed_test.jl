using Distances
import LinearAlgebra
using LinearAlgebra
BLAS.set_num_threads(1)

function cosine_theirs(states)
    Distances.pairwise(Distances.CosineDist(), states, states)
end

function cosine_mine(states, states2)
    states = states .* inv.(sqrt.(sum(abs2, states; dims=1)))
    states2 = states2 .* inv.(sqrt.(sum(abs2, states2; dims=1)))
    # similarity = states' * states
    # return 1 .- similarity
    # result = Matrix{Float32}(undef, size(states, 2), size(states, 2))
    # @inbounds for i in 1:size(states, 2)
    #     for j in i:size(states, 2)
    #         s = 1.0f0 - LinearAlgebra.dot(@view(states[:, i]), @view(states[:, j]))
    #         result[i, j] = s
    #     end
    #     result[i, i] = 0.0
    # end

    # return result
    return 1.0f0 .- states2' * states
end

function euclidean_theirs(states)
    Distances.pairwise(Distances.Euclidean(), states, states)
end

function euclidean_mine(states1, states2)
    n1, n2 = size(states1, 2), size(states2, 2)
    features = size(states1, 1)
    result = Matrix{Float32}(undef, n1, n2)

    @fastmath @inbounds for i in 1:n1
        for j in 1:n2
            s = 0.0f0
            @simd for k in 1:features
                # s += (states1[k, i] - states2[k, j]) ^ 2
                x = states1[k, i]
                y = states2[k, j]
                s += 2 * x * y - x * x - y * y
            end
            result[i, j] = s
        end
    end
    return result
end

function city_block_mine(states1, states2)
    result = Matrix{Float32}(undef, size(states1, 2), size(states2, 2))

    @inbounds for i in 1:size(states1, 2)
        for j in 1:size(states2, 2)
            s = 0.0f0
            @simd for k in 1:size(states1, 1)
                s += abs(states1[k, i] - states2[k, j])
            end
            s = (s)
            result[i, j] = s
        end
    end

    return result
end

function tests()
    random_states = rand(Float32, 300, 100)
    random_states2 = rand(Float32, 300, 100)
    n = 1_000
    t = nothing


    view_vector1 = @view random_states[:, 1]
    view_vector2 = @view random_states[:, 2]

    @time for _ in 1:n
        # t = Distances.cosine_dist(view_vector1, view_vector2)
        t = Distances.CosineDist()(view_vector1, view_vector2)
        t = Distances.Euclidean()(view_vector1, view_vector2)
    end

    @time for _ in 1:sqrt(n)
        random_states = random_states .* inv.(sqrt.(sum(abs2, random_states; dims=1)))
    end
    view_vector1 = @view random_states[:, 1]
    view_vector2 = @view random_states[:, 2]
    @time for _ in 1:n
        t = 1.0f0 - LinearAlgebra.dot(view_vector1, view_vector2)
    end

    println("cosine")
    @time for _ in 1:n
        t = Distances.pairwise(Distances.CosineDist(), random_states, random_states2)
    end

    println("cosine_mine")
    @time for _ in 1:n
        t = cosine_mine(random_states, random_states2)
    end

    println("euclidean")
    distance_metric = Distances.Euclidean()
    @time for _ in 1:n
        t = Distances.pairwise(distance_metric, random_states, random_states2)
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