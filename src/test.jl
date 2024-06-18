using Revise

include("neural_network/NeuralNetwork.jl")
using .NeuralNetwork
using Flux

nn = NeuralNetwork.MLP_NN(input_size=10, output_size=10, hidden_layers=3, hidden_neurons=256, activation_function=:relu, last_activation_function=[(:softmax, 5), (:softmax, 5)])#(x) -> vcat(softmax(@view x[1:5, :]), softmax(@view x[6:10, :])))#[(:sigmoid, 5), (:tanh, 5)])#:sigmoid)

input = rand(Float32, 10, 20)
true_output = rand(Float32, 10, 20)
true_output = vcat(softmax(@view true_output[1:5, :]), softmax(@view true_output[6:10, :]))

output = NeuralNetwork.predict(nn, input)
display(true_output)
display(output)

learn!(nn, input, true_output, Flux.kldivergence, epochs=40, batch_size=32, learning_rate=0.001)
output = NeuralNetwork.predict(nn, input)
println("done")

display(true_output)
display(output)
