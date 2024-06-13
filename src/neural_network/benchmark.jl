using MKL
using Flux
using BenchmarkTools
using Random
using LinearAlgebra

include("MLP.jl")
using .MLP

# Set the number of BLAS threads to 1
BLAS.set_num_threads(1)
println(BLAS.get_config())

# Check if the setting is applied
println("Number of BLAS threads: ", LinearAlgebra.BLAS.get_num_threads())
println("Number of BLAS threads: ", BLAS.get_num_threads())



# Create input data
input_data = rand(Float32, 10, 5)  # 5 samples with 10 features each

# Instantiate the model
model = MLP_NN(;
    input_size=10,
    output_size=3,
    hidden_layers=2,
    hidden_neurons=64,
    activation_function=:relu,
    last_activation_function=:tanh#[(:softmax, 2), (:tanh, 1)]#[(:relu, 2), (:tanh, 1)]  # [(:softmax, 2), (:tanh, 1)]  # :none
)

# Measure performance of the model


Activation = [(:tanh, 2), (:tanh, 1)]

model2 = MLP_NN(;
input_size=10,
output_size=3,
hidden_layers=2,
hidden_neurons=64,
activation_function=:relu,
last_activation_function=:softmax#[(:tanh, 3)]#[(:softmax, 2), (:tanh, 1)]#[(:relu, 2), (:tanh, 1)]  # [(:softmax, 2), (:tanh, 1)]  # :none
)

model3 = MLP_NN(;
input_size=10,
output_size=3,
hidden_layers=2,
hidden_neurons=64,
activation_function=:relu,
last_activation_function=[(:softmax, 2), (:tanh, 1)]#[(:softmax, 2), (:tanh, 1)]#[(:relu, 2), (:tanh, 1)]  # [(:softmax, 2), (:tanh, 1)]  # :none
)

function benchmark_model(model, input_data)
    for _ in 1:100_000
        predict(model, input_data)
        # println(predict(model, input_data))
        # exit
    end
end

# for _ in 1:100
#     predict(model, input_data)
# end

@time benchmark_model(model, input_data)
@time benchmark_model(model2, input_data)
@time benchmark_model(model3, input_data)
display(predict(model, input_data))
# using Plots
# plot(1:10, 1:10)