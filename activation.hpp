#ifndef SOMA_ACTIVATION_HPP
#define SOMA_ACTIVATION_HPP

#include <string>
#include <algorithm>
#include "./matrix.hpp"

namespace Soma::Activation {
	// Sigmoid activation function
	// σ(x) = 1 / (1 + e^-x)
	inline Matrix sigmoid(Matrix input) {
		input.forEach([](double* x, int, int) {
			*x = 1.0 / (1.0 + std::exp(-*x));
		});
		return input;
	}
	// Derivative of sigmoid activation function
	// σ'(x) = y * (1 - y)
	inline Matrix sigmoidDerivative(Matrix output) {
		return output * (1.0 - output);
	}
	// ReLU activation function
	// ReLU(x) = max(0, x)
	inline Matrix relu(Matrix input) {
		input.forEach([](double* x, int, int) {
			*x = std::max(0.0, *x);
		});
		return input;
	}
	// Derivative of ReLU activation function
	// ReLU'(x) = 1 if y > 0 else 0
	inline Matrix reluDerivative(Matrix output) {
		output.forEach([](double* y, int, int) {
			*y = *y > 0.0 ? 1.0 : 0.0;
		});
		return output;
	}
	// Hyperbolic tangent activation function
	// tanh(x) = (e^x-e^-x)/(e^x+e^-x)
	inline Matrix tanh(Matrix input) {
		input.forEach([](double* x, int, int) {
			*x = std::tanh(*x);
		});
		return input;
	}
	// Derivative of hyperbolic tangent activation function
	// Let y = tanh(x)
	// tanh'(x) = 1 - y^2
	inline Matrix tanhDerivative(Matrix output) {
		return 1 - (output ^ 2);
	}
	// Softmax activation function
	// softmax(X)_i = e^(X_i) / sum_j=1^N e^(X_j)
	inline Matrix softmax(Matrix input) {
		double sum = 0;
		double max = input.max();
		input.forEach([&sum, max, &input](double* x, int, int) {
			*x = std::exp(*x - max); // Subtract max for numerical stability while maintaining output
			sum += *x;
		});
		return input / sum;
	}
	// Derivative of softmax activation function
	// This derivative is special as it directly gives the p.d. of the loss w.r.t. the input
	// This derivative is a simplification of the actual derivative which is a Jacobian matrix
	// Let y_i = softmax(X)_i and dy be the p.d. of the loss w.r.t. to y
	// softmax'(X) = y(dy - s) where s = y^T . dy
	inline Matrix softmaxDerivative(Matrix output, Matrix dy) {
		double s = Matrix::dot(output.transpose(), dy)[0][0];
		return output * (dy - s);
	}
}

// Activation functions are network attributes and need to be specified in the network
// The strings in this namespace are used to identify the activation functions of the network for saving and loading
namespace Soma::ActivationType {
	const std::string Sigmoid = "sigmoid";
	const std::string ReLU = "relu";
	const std::string Tanh = "tanh";
	const std::string Softmax = "softmax";
}

#endif
