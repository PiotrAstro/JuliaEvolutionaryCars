module TestM
using MKL
import LoopVectorization
import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
import Test
using BenchmarkTools

struct TestNN{M<:Val}
    encoded_exemplars_normlized::Matrix{Float32}
    encoded_exemplars_normalized_transposed::Matrix{Float32}
    translation::Vector{Int}
    translation_cont::Matrix{Float32}
    actions_number::Int
end

function TestNN(encoded_exemplars::Matrix{Float32}, m::Int)
    encoded_exemplars_normlized = normalize_unit(encoded_exemplars)
    encoded_exemplars_normalized_transposed = collect(encoded_exemplars_normlized')
    return TestNN{Val{m}}(encoded_exemplars_normlized, encoded_exemplars_normalized_transposed, collect(1:size(encoded_exemplars, 2)), Matrix{Float32}(undef, 2, 2), m)
end
function TestNN(encoded_exemplars::Matrix{Float32}, m::Int, tranlation, translation_cont, n_actions)
    encoded_exemplars_normlized = normalize_unit(encoded_exemplars)
    encoded_exemplars_normalized_transposed = collect(encoded_exemplars_normlized')
    return TestNN{Val{m}}(encoded_exemplars_normlized, encoded_exemplars_normalized_transposed, tranlation, translation_cont, n_actions)
end
global const Episol::Float32 = Float32(1e-6)

function calculate_distance_normal(test_nn::TestNN{M}, new_exemplars) where M
    # new_exemplars = normalize_unit(new_exemplars)
    normalize_unit!(new_exemplars)
    distances = test_nn.encoded_exemplars_normalized_transposed * new_exemplars
    @inbounds @fastmath for i in eachindex(distances)
        value = 1.0f0 - distances[i]
        distances[i] = ifelse(value < Episol, Episol, value)
    end

    return distances
end

function calculate_distance_mine(test_nn::TestNN{M}, new_exemplars) where M
    # distances = Matrix{Float32}(undef, size(test_nn.encoded_exemplars_normlized, 2), size(new_exemplars, 2))
    # @fastmath for col in axes(new_exemplars, 2)
    #     scaling_factor = 0.0f0
    #     LoopVectorization.@turbo for i in axes(new_exemplars, 1)
    #         scaling_factor += new_exemplars[i, col] * new_exemplars[i, col]
    #     end
    #     scaling_factor = 1.0f0 / sqrt(scaling_factor)
    #     LoopVectorization.@turbo for row in axes(test_nn.encoded_exemplars_normlized, 2)
    #         value = 1.0f0
    #         for i in axes(test_nn.encoded_exemplars_normlized, 1)
    #             value -= test_nn.encoded_exemplars_normlized[i, row] * new_exemplars[i, col] * scaling_factor
    #         end
    #         distances[row, col] = value
    #     end
    # end
    # @inbounds for i in eachindex(distances)
    #     distances[i] = ifelse(distances[i] < Episol, Episol, distances[i])
    # end
    # return distances

    # new_exemplars = normalize_unit(new_exemplars)
    normalize_unit!(new_exemplars)
    distances = Matrix{Float32}(undef, size(test_nn.encoded_exemplars_normlized, 2), size(new_exemplars, 2))
    LoopVectorization.@turbo for col in axes(new_exemplars, 2), row in axes(test_nn.encoded_exemplars_normlized, 2)
        value = 1.0f0
        for i in axes(test_nn.encoded_exemplars_normlized, 1)
            value -= test_nn.encoded_exemplars_normlized[i, row] * new_exemplars[i, col] #* scaling_factor
        end
        distances[row, col] = value
    end
    @inbounds for i in eachindex(distances)
        distances[i] = ifelse(distances[i] < Episol, Episol, distances[i])
    end
    return distances
end

function normalize_unit(x::Matrix{Float32}) :: Matrix{Float32}
    copied = Base.copy(x)
    normalize_unit!(copied)
    return copied
end

using LoopVectorization
function normalize_unit!(x::Matrix{Float32})
    # for col in eachcol(x)
    #     LinearAlgebra.normalize!(col)
    # end

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

function test_distances()
    exemplars = rand(Float32, 100, 1000)
    test_nn = TestNN(exemplars, 2)
    new_exemplars = rand(Float32, 100, 10)

    normal_result = calculate_distance_normal(test_nn, new_exemplars)
    mine_result = calculate_distance_mine(test_nn, new_exemplars)
    t = Test.@test normal_result ≈ mine_result
    display(t)

    # b = BenchmarkTools.@benchmark collect(($exemplars)')
    # display(b)

    b = BenchmarkTools.@benchmark calculate_distance_normal($test_nn, $new_exemplars)
    display(b)

    b = BenchmarkTools.@benchmark calculate_distance_mine($test_nn, $new_exemplars)
    display(b)
end


function distance_euclidean(x::Vector{Float32}, y::Vector{Float32})::Float32
    summed = 0.0f0
    @inbounds @simd for i in eachindex(x)
        summed += (x[i] - y[i]) ^ 2
    end
    return sqrt(summed)
end






# predictions

function predict_all(nn::TestNN{Val{M_INT}}, distances::Matrix{Float32})::Matrix{Float32} where {M_INT}
    result_matrix = zeros(Float32, nn.actions_number, size(distances, 2))
    @inbounds for (i, result_col) in enumerate(eachcol(result_matrix))
        @fastmath @simd for exemplar_id in eachindex(nn.translation)
            result_col[nn.translation[exemplar_id]] += (1.0f0 / distances[exemplar_id, i]) ^ M_INT
        end
        result_col ./= sum(result_col)
    end
    return result_matrix
end

function predict_all_mine(nn::TestNN{Val{M_INT}}, distances::Matrix{Float32})::Matrix{Float32} where {M_INT}
    result_matrix = zeros(Float32, nn.actions_number, size(distances, 2))
    # LoopVectorization.@turbo 
    @inbounds @fastmath for exemplar_id in axes(nn.translation, 1)
        @simd for col in axes(distances, 2)
            result_matrix[nn.translation[exemplar_id], col] += (1.0f0 / distances[exemplar_id, col]) ^ M_INT
        end
    end
    for result_col in eachcol(result_matrix)
        @fastmath result_col ./= sum(result_col)
    end
    return result_matrix
end

function predict_all_cont(nn::TestNN{Val{M_INT}}, distances::Matrix{Float32})::Matrix{Float32} where {M_INT}
    result_matrix = zeros(Float32, nn.actions_number, size(distances, 2))
    @inbounds for (i, result_col) in enumerate(eachcol(result_matrix))
        @fastmath @simd for exemplar_id in axes(nn.translation_cont, 2)
            member = (1.0f0 / distances[exemplar_id, i]) ^ M_INT
            for row in 1:nn.actions_number
                result_col[row] += nn.translation_cont[row, exemplar_id] * member
            end
        end
        result_col ./= sum(result_col)
    end
    return result_matrix
end

function predict_all_cont_mine(nn::TestNN{Val{M_INT}}, distances::Matrix{Float32})::Matrix{Float32} where {M_INT}
    result_matrix = zeros(Float32, nn.actions_number, size(distances, 2))
    @inbounds for (i, result_col) in enumerate(eachcol(result_matrix))
        LoopVectorization.@turbo for exemplar_id in axes(nn.translation_cont, 2)
            member = (1.0f0 / distances[exemplar_id, i]) ^ M_INT
            for row in 1:nn.actions_number
                result_col[row] += nn.translation_cont[row, exemplar_id] * member
            end
        end
        result_col ./= sum(result_col)
    end
    return result_matrix
end

function test_final()
    latent = 32
    exemplars_n = 40
    new_exemplars_n = 5
    exemplars = rand(Float32, latent, exemplars_n)
    n_actions = 9
    translation = rand(1:n_actions, exemplars_n)
    translation_cont = rand(Float32, n_actions, exemplars_n)
    test_nn = TestNN(exemplars, 2, translation, translation_cont, n_actions)
    new_exemplars = rand(Float32, latent, new_exemplars_n)
    distances = calculate_distance_normal(test_nn, new_exemplars)

    println("\n\n\n\n\n\ndistances")
    normal_result = calculate_distance_normal(test_nn, new_exemplars)
    mine_result = calculate_distance_mine(test_nn, new_exemplars)
    t = Test.@test normal_result ≈ mine_result
    display(t)
    b = BenchmarkTools.@benchmark calculate_distance_normal($test_nn, $new_exemplars)
    display(b)

    b = BenchmarkTools.@benchmark calculate_distance_mine($test_nn, $new_exemplars)
    display(b)
    

    println("translations")
    mine_normal = predict_all(test_nn, distances)
    mine_new = predict_all_mine(test_nn, distances)
    t = Test.@test mine_normal ≈ mine_new
    display(t)

    b = BenchmarkTools.@benchmark predict_all($test_nn, $distances)
    display(b)

    b = BenchmarkTools.@benchmark predict_all_mine($test_nn, $distances)
    display(b)

    # cont

    println("translations cont")
    mine_normal = predict_all_cont(test_nn, distances)
    mine_new = predict_all_cont_mine(test_nn, distances)
    t = Test.@test mine_normal ≈ mine_new
    display(t)

    b = BenchmarkTools.@benchmark predict_all_cont($test_nn, $distances)
    display(b)

    b = BenchmarkTools.@benchmark predict_all_cont_mine($test_nn, $distances)
    display(b)
end

end

import .TestM
# TestM.test_distances()
TestM.test_final()
