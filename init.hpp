#ifndef SOMA_INIT_HPP
#define SOMA_INIT_HPP

#include <chrono>
#include <random>

#include "layer.hpp"
#include "network.hpp"

namespace Soma::Initialization {
	// Helper functions for initializations
	inline unsigned int getSeed() {
		return static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	}
	// Weight initialization functions

	// Uniform Xavier initialization
	// Initialize weights uniformly across +- sqrt(6/(in + out))
	inline void xavierUniform(Network& nn, unsigned int seed = getSeed()) {
		std::mt19937 gen(seed);
		for (int i = 0; i < nn.depth; i++) {
			Layers::DenseLayer* layer = dynamic_cast<Layers::DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			double limit = std::sqrt(6.0 / (layer->inCount + layer->outCount));
			std::uniform_real_distribution<double> dis(-limit, limit);
			layer->W.forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}
	// Normal Xavier initialization
	// Initialize weights with a normal distribution with mean of 0 and standard deviation of sqrt(2/(in + out))
	inline void xavierNormal(Network& nn, unsigned int seed = getSeed()) {
		std::mt19937 gen(seed);
		for (int i = 0; i < nn.depth; i++) {
			Layers::DenseLayer* layer = dynamic_cast<Layers::DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			double stddev = std::sqrt(2.0 / (layer->inCount + layer->outCount));
			std::normal_distribution<double> dis(0.0, stddev);
			layer->W.forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}
	// Uniform He initialization
	// Initialize weights uniformly across +- sqrt(6/in)
	inline void heUniform(Network& nn, unsigned int seed = getSeed()) {
		std::mt19937 gen(seed);
		for (int i = 0; i < nn.depth; i++) {
			Layers::DenseLayer* layer = dynamic_cast<Layers::DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			double limit = std::sqrt(6.0 / layer->inCount);
			std::uniform_real_distribution<double> dis(-limit, limit);
			layer->W.forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}
	// Normal He initialization
	// Initialize weights with a normal distribution with mean of 0 and standard deviation of sqrt(2/in)
	inline void heNormal(Network& nn, unsigned int seed = getSeed()) {
		std::mt19937 gen(seed);
		for (int i = 0; i < nn.depth; i++) {
			Layers::DenseLayer* layer = dynamic_cast<Layers::DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			double stddev = std::sqrt(2.0 / layer->inCount);
			std::normal_distribution<double> dis(0.0, stddev);
			layer->W.forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}
	// SIREN parameter initialization
	// Initialize first SIREN layer omega0 with 30.0 by default and parameters uniformly across +- 1/in
	// Initialize other SIREN layer parameters uniformly across +- sqrt(6/in)/omega0
	inline void SIRENInit(Network& nn, double omega0 = 30.0, unsigned int seed = getSeed()) {
		std::mt19937 gen(seed);
		for (int i = 0; i < nn.depth; i++) {
			Layers::SIRENLayer* layer = dynamic_cast<Layers::SIRENLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-SIRENLayers

			double limit = std::sqrt(6.0 / layer->inCount) / layer->omega0;
			if (i == 0) {
				limit = 1.0 / layer->inCount;
				layer->omega0 = omega0;
			}
			std::uniform_real_distribution<double> dis(-limit, limit);
			for (Matrix& param : layer->params) {
				param.forEach([&dis, &gen](double *val, int, int) {
					*val = dis(gen);
				});
			}
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}

	// Bias initialization functions

	// Initialize biases to a constant
	inline void constantBias(Network& nn, double constant) {
		for (int i = 0; i < nn.depth; i++) {
			Layers::DenseLayer* layer = dynamic_cast<Layers::DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			layer->B.forEach([constant](double *val, int, int) {
				*val = constant;
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}
}

// Initializers are standalone functions and not network attributes
// Therefore, the initialization functions being used need not be specified in the network

#endif
