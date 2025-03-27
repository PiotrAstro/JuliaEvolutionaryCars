# JuliaEvolutionaryCars

A Julia package that explores evolutionary methods for solving reinforcement learning environments, specifically focusing on car control and navigation tasks. The project combines Neural Networks' signal processing capabilities with Genetic Algorithms' exploration capabilities to create efficient learning systems.

Currently there is only one environment - Car Racing, but this framework can work with any environment! I chose unfortunate name.

Some might ask: Why Julia? Why not Python?
Well, I have already tried doing it with Python: https://github.com/PiotrAstro/Evolutionary_Cars
I even created my own Neural Network framework in Cython, along with some Cython environments to work around the Global Interpreter Lock (GIL) - and it worked! However, when I needed to implement more neural network functionality, I decided it was finally time to switch languages and leave those previous challenges behind. Julia offered the perfect solution to the performance issues I had been struggling with in Python.

## Features

### Core Components
- **Neural Networks**: Advanced neural network implementations for policy representation
- **Evolutionary Algorithms**: Multiple constantly changing algorithms :)
- **Environments**: Custom environments, currently just one car environment
- **Utilities**: Additional features, many clustering algorithms etc.

- **Scripts**: Multiple Julia performance tests, scripts running calculations, configuraiton file etc.

### Key Capabilities
- Hybrid learning approach combining neural networks and genetic algorithms
- Efficient state grouping and optimization
- Distributed computing among cluster when conducting comparison of parameters
- Advanced visualization and analysis tools

## Project Structure

```
.
├── src/                    # Source code
│   ├── NeuralNetwork/     # Neural network implementations
│   ├── EvolutionaryComputation/  # Evolutionary algorithms
│   ├── Environments/      # RL environments
│   └── utils/            # Utility functions
├── scripts/               # Execution and analysis scripts
│   ├── interpret_results/ # Results analysis and plotting
│   └── tmp_tests/        # Performance and functionality tests
├── data/                  # Data storage
└── log/                   # Logging directory
```

## Dependencies

The project uses several key Julia packages:
- **Machine Learning**: Flux.jl for neural networks
- **Optimization**: MKL for optimized linear algebra
- **Data Processing**: DataFrames.jl, CSV.jl
- **Visualization**: Plots.jl, StatsPlots.jl
- **Performance**: BenchmarkTools.jl, Profile.jl
- **Utilities**: Dates.jl, Logging.jl, Statistics.jl

## Installation

1. Ensure you have Julia installed (version 1.11 or higher recommended)
2. Clone this repository
3. Install dependencies:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

## Usage

### Running Experiments
Remember to set correct environment
```bash
julia --project=. scripts/run_one.jl
```
or
```bash
julia --project=. scripts/params_tests/run_params_tests.jl
```
Configure you cluster before running parameters tests.

### Analyzing Results
```bash
julia --project=. scripts/interpret_results/plotting_interpret.jl
```

## Performance Optimization

Main performance optimization was changing language from Python to Julia, all of my main problems are gone :)
Now I do not have to care about:
- Multithreading
- performance of hot code in general (it is hihly optimized now, comparable to fully compiled languages)

The project includes several optimizations:
- It is in Julia :)
- Ability to use huge cluster for calculation (tested with 6 computers, 80 threads in total)
- Multithreading
- Efficient state grouping algorithms
- MKL integration for fast linear algebra operations

## Author

Piotr Zatwarnicki (piotr.l.zatwarnicki@gmail.com)

## Future Work

- GPU adaptation
- Implementation of more Reinforcement Learning environments, like Atari, Humanoid etc.

Stay tuned for more updates and publications!