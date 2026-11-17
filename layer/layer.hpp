#ifndef SOMA_LAYER_LAYER_HPP
#define SOMA_LAYER_LAYER_HPP

#include <fstream>
#include <memory>
#include <vector>

#include "../matrix.hpp"

namespace Soma::Layers {
	class Layer {
	public:
		int inCount = 0, outCount = 0; // Number of input and output neurons for the layer

		// Optional parameters and gradients for training
		std::vector<std::reference_wrapper<Matrix>> params;
		std::vector<Matrix> grads;
		// Optional last input and output storage for backpropagation
		Matrix lastInput, lastOutput;

		Layer(int inCount = 0, int outCount = 0) : inCount(inCount), outCount(outCount) {}

		virtual ~Layer() = default;
		// Returns an output without setting last input or output
		virtual Matrix run(const Matrix& x) = 0;
		// Returns an output and sets last input and/or output
		virtual Matrix forward(const Matrix& x) = 0;
		// Sets gradients and returns error for input
		virtual Matrix backward(const Matrix& dy) = 0;

		// Save layer data to the file stream
		virtual void save(std::ofstream& out) = 0;
		// Factory loader
		static std::unique_ptr<Layer> load(std::ifstream& in);
	};
}

#endif
