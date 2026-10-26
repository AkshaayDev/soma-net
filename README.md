# soma-net

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  1. [Network features](#1-network-features)
  2. [NNMatrix features](#2-nnmatrix-features)
  3. [Layers](#3-layers)
  4. [Initializations](#4-initializations)
  5. [Activation functions](#5-activation-functions)
  6. [Loss functions](#6-loss-functions)
  7. [Optimizers and Their Hyperparameters](#7-optimizers-and-their-hyperparameters)
  8. [Trainers](#8-trainers)
- [Usage](#usage)
  1. [Cloning and Including](#1-cloning-and-including)
  2. [Creating and Configuring the Network](#2-creating-and-configuring-the-network)
  3. [Running the network](#3-running-the-network)
  4. [Saving and Loading](#4-saving-and-loading)
  5. [Training](#5-training)
- [Examples](#examples)

## Overview

Soma is a minimal C++ library for **creating simple neural networks** from scratch.
Apart from standard C++ libraries, it does not have any external dependencies.
This project is experimental and for educational purposes.

> Disclaimer: Although the neural network library headers do not have any external dependencies, some of the examples in the `/examples` folder may have external dependencies (like raylib.h or stb_image.h)

## Features

- Minimal purpose-built `NNMatrix` matrix class
- Feed-forward networks with `NeuralNetwork` class
- Network initialization, activation functions, loss functions
- Forward propagation and backpropagation
- Network saving and loading with a file stream
- Customizable trainer and optimizer objects

### 1. Network features

- Layer adding
- Set loss function
- Averaging gradients
- Running the network
- Forward and backward propagation
- Saving and loading network data (optionally with optimizer data)

### 2. NNMatrix features

- `std::vector<std::vector<double>>` constructor
- `rows` and `cols` constructor
- `rows()` and `cols()` getter functions
- Static matrix printing
- Static matching size checking
- Static `fromVector(std::vector<double>)` helper
- Static `fromScalar(double)` helper
- Resize rows and columns
- `forEach(const std::function<void(double*, int, int)>& func)` function to apply `func` to each element, passing element pointer, row and column as arguments
- `fill(double)` function to fill the matrix with the value
- Check for `nan`s
- Scalar and element-wise addition, subtraction, multiplication and division
- Scalar exponentiation
- Access data directly with `NNMatrix[row]`
- Static dot product
- Transpose of matrix
- Maximum value of matrix
- Element sum

### 3. Layers

- DenseLayer
- ActivationLayer
- SIRENLayer

### 4. Initializations

- Xavier (Normal/Uniform)
- He (Normal/Uniform)
- SIREN weights
- Constant biases

### 5. Activation functions

- Sigmoid
- ReLU
- tanh
- Softmax

### 6. Loss functions

- Mean Squared Error
- Categorical Cross Entropy

### 7. Optimizers and Their Hyperparameters

- Gradient Descent
  - `learningRate` = `0.001`
- Momentum
  - `learningRate` = `0.001`
  - `beta` = `0.9`
- Adam
  - `learningRate` = `0.001`
  - `beta1` = `0.9`
  - `beta2` = `0.999`
  - `epsilon` = `1e-8`

### 8. Trainers

Attaches an optimizer and training data batch to the network and handles training data.

- Iteration and epoch callbacks
- Sample size
- Data shuffling

## Usage

### 1. Cloning and Including

Clone this repository:

```bash
git clone https://github.com/akshaaydev/soma-net.git
```

Then, include from the main neural network header `soma.hpp`.

```c++
#include "/soma-net/soma.hpp"
```

### 2. Creating and Configuring the Network

To create a network, define a `NeuralNetwork` object.

```c++
NeuralNetwork nn;
```

To set the network architecture, use the `addLayer()` function.

```c++
nn.addLayer<DenseLayer>(2, 2); // This sets a dense layer that takes in and outputs 2 neurons
nn.addLayer<ActivationLayer>(2, ActivationType::Sigmoid); // Applies sigmoid activation to the 2 neurons
```

To initialize the parameters of the network, use one of the functions from the `Initialization` namespace.

```c++
Initialization::xavierNormal(nn);
```

To set the loss function, use `setLossFunction()`.

```c++
// This sets the loss function to Mean Squared Error(MSE)
nn.setLossFunction(LossType::MSE);
```

### 3. Running the Network

To run the network, use `run()`.

```c++
// This does not set anything and returns the NNMatrix output of the network
NNMatrix predicted = nn.run(input);
```

To forward propagate an input and set relevant last inputs and outputs for each layer, use `forwardPropagation()`.

```c++
// This performs forward propagation with the input and returns the network output
NNMatrix input;
// Define `input` here
NNMatrix predicted = nn.forwardPropagation(input);
```

To set parameter gradients, use `backwardPropagation()`.

> Important: `forwardPropagation()` always has to be called to update activations before `backwardPropagation()`.
> However, propagation is handled by the network during training.

```c++
NNMatrix real;
// Define `real` here
nn.backwardPropagation(predicted, real);
```

### 4. Saving and Loading

To save network parameters and architecture, use the `save()` and `load()` functions.
To save the optimizer data, call `save()` with an `Optimizer*` pointer to the optimizer.
To load the optimizer data, call `load()` with a `std::unique_ptr<Optimizer>` pointer to the optimizer. If there is no optimizer data found in the input file stream, it will be ignored.

```c++
MomentumOptimizer mom(nn);
// Write the neural network data to a file `out.dat` with binary encoding
std::ofstream out("out.dat", std::ios::binary);
nn.save(out, &mom); // Save the network data to that file
out.close(); // Close the file
```

```c++
MomentumOptimizer mom(nn);
// Read the neural network data from a file `in.dat` with binary encoding
std::ifstream in("in.dat", std::ios::binary);
nn.load(in, std::make_unique<MomentumOptimizer>(mom)); // Load the network data from that file
in.close(); // Close the file
```

### 5. Training

Before training the network, an optimizer object must be created to specify the type and hyperparameters of optimization. The optimizer type can be any derived class of the `Optimizer` class.

```c++
GradientDescentOptimizer gd(nn) // Attaches a network to `gd` with default parameters
GradientDescentOptimizer gd(nn, 15) // Attaches a network to `gd` with learning rate 15
```

The optimizer constructor must include the network object and optionally the hyperparameters(See Optimizer features for their hyperparameters).

To train the network, a trainer object must be created and initialized with the network, optimizer and batch.
The batch is a `std::vector` of samples which is a `std::pair` of the input and output `NNMatrix`.

```c++
std::vector<std::pair<NNMatrix, NNMatrix>> batch;
// Example sample that maps {{0},{0}} to {{0}}
batch.push_back(std::make_pair(NNMatrix::fromVector({0,0}), NNMatrix::fromScalar(0.0))); // 0 ^ 0 = 0

Trainer trainer(nn, gd, batch);
```

> Note: The network and batch in the constructor are passed by reference.
> If batch shuffling is enabled, it will modify the original batch too.
> To prevent this, you can copy the batch into a new variable or disable shuffling (See below).

Callbacks for iterations and epochs can be set like this:

```c++
trainer.iterationCallback = []() { std::cout << "Iteration " << nn.iterationsTrained << "\n"; };
trainer.epochCallback = []() { std::cout << "Epoch " << nn.epochsTrained << "\n"; };
```

The batch is divided into samples to update parameters based on each sample's derivatives.
The sample size is `-1` by default, meaning the whole batch is trained in an iteration.
By default, the batch is shuffled before before every epoch but this can be disabled.

```c++
trainer.sampleSize = 128;
trainer.enableShuffling = false;
```

Finally call the `train()` method with the number of epochs.

```c++
trainer.train(100);
```

## Examples

For example usages, refer to [soma-net-examples](https://github.com/akshaaydev/soma-net-examples.git) repository.
