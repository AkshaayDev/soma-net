#ifndef SOMA_NETWORK_HPP
#define SOMA_NETWORK_HPP

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer.hpp"
#include "loss.hpp"

namespace Soma::Optimizers { class Optimizer; }

namespace Soma {

class Network {
public:
	static const std::uint32_t SIGNATURE = 0x534f4d41; // "SOMA", Used for validating data files
	std::vector<std::unique_ptr<Layers::Layer>> layers;
	// Represents the structure of the neural network (How many neurons is in each layer)
	int depth = 0; // Number of layers
	int iterationsTrained = 0, epochsTrained = 0;

	// Averaged gradients of each layer
	std::vector<std::vector<Matrix>> avgGrads;

	// Loss function for the network
	std::string lossFnName;
	std::function<double(Matrix, Matrix)> lossFn;
	std::function<Matrix(Matrix, Matrix)> lossFnDerivative;

	// Setters

	// Add a layer to the network (Usage: nn.addLayer<LayerType>(args); )
	template<typename LayerType, typename... Args>
	void addLayer(Args&&... args) {
		layers.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
		avgGrads.push_back(layers.back()->grads);
		depth++;
	}

	// Set the loss function of the network
	// Pass LossType as argument
	void setLossFunction(std::string loss) {
		if (loss == LossType::MSE) {
			lossFn = Loss::MSE;
			lossFnDerivative = Loss::MSEDerivative;
		} else if (loss == LossType::CCE) {
			lossFn = Loss::CCE;
			lossFnDerivative = Loss::CCEDerivative;
		} else throw std::runtime_error("Unknown loss function ('" + loss + "')");
		lossFnName = loss;
	}

	// Accumulate and average the gradients for each sample in the batch
	void averageGrads(std::vector<std::pair<Matrix, Matrix>> batch) {
		for (int i = 0; i < depth; i++) {
			for (Matrix& avgGrad : avgGrads[i]) {
				avgGrad.fill(0);
			}
		}
		for (std::pair<Matrix, Matrix> sample : batch) {
			Matrix predicted = forwardPropagation(sample.first);
			backwardPropagation(predicted, sample.second);
			for (int i = 0; i < depth; i++) {
				for (int j = 0; j < layers[i]->grads.size(); j++) {
					avgGrads[i][j] = avgGrads[i][j] + layers[i]->grads[j];
				}
			}
		}
		for (int i = 0; i < depth; i++) {
			for (Matrix& avgGrad : avgGrads[i]) {
				avgGrad = avgGrad / batch.size();
			}
		}
	}

	// Performs a feed forward without storing inputs or outputs
	Matrix run(Matrix input) {
		if (layers.empty()) throw std::runtime_error("Cannot run an empty network");
		for (auto& layer : layers) {
			input = layer->run(input);
		}
		return input;
	}
	// Sets layer inputs and outputs after forward propagation of an input and returns network output
	Matrix forwardPropagation(Matrix input) {
		if (layers.empty()) throw std::runtime_error("Cannot forward propagate through an empty network");
		for (auto& layer : layers) {
			input = layer->forward(input);
		}
		return input;
	}
	// Sets the layer gradients (partial derivatives of the loss with respect to its parameters)
	// Note: forward propagation has to be called first and its recommended to pass its return value as `predicted`
	void backwardPropagation(Matrix predicted, Matrix real) {
		if (layers.empty()) throw std::runtime_error("Cannot backward propagate through an empty network");
		Matrix dy = lossFnDerivative(predicted, real);
		for (int i = depth - 1; i >= 0; i--) {
			dy = layers[i]->backward(dy);
		}
	}

	// Save the parameters, architecture and optionally moments to an output file stream
	// `opt` is the a pointer to the optimizer for saving moments (leave empty to not save moments)
	void save(std::ofstream& out, Optimizers::Optimizer* opt = nullptr);

	// Load the parameters, architecture and optionally moments from an input file stream
	// `opt` is the a pointer to the optimizer for loading moments (leave empty to not load moments)
	void load(std::ifstream& in, std::unique_ptr<Optimizers::Optimizer> opt = nullptr);
};

}

#endif
