#ifndef SOMA_LOSS_HPP
#define SOMA_LOSS_HPP

#include <string>
#include "matrix.hpp"

namespace Soma::Loss {
	const double epsilon = 1e-12;
	// Mean Squared Error
	// MSE = 1/n * ∑(p_i - r_i)^2
	inline double MSE(NNMatrix predicted, NNMatrix real) {
		return ((predicted - real) ^ 2.0).sum() / real.rows();
	}
	// Derivative of Mean Squared Error
	// MSE' = 2/n * (p_i - r_i)
	inline NNMatrix MSEDerivative(NNMatrix predicted, NNMatrix real) {
		return 2.0 / real.rows() * (predicted - real);
	}
	// Categorical Cross Entropy Loss
	// CCE = - ∑ r_i log(p_i + ε)
	inline double CCE(NNMatrix predicted, NNMatrix real) {
		double sum = 0;
		predicted.forEach([&sum, &real](double *val, int i, int j) {
			sum -= real[i][j] * std::log(*val + epsilon); // epsilon to avoid log(0)
		});
		return sum;
	}
	// Derivative of Categorical Cross Entropy Loss
	// CCE' = - r_i / (p_i + ε)
	inline NNMatrix CCEDerivative(NNMatrix predicted, NNMatrix real) {
		return -real / (predicted + epsilon); // epsilon to avoid / 0
	}
}

// Loss functions are network attributes and need to be specified in the network
// The strings in this namespace are used to identify the loss function in the network for saving and loading
namespace Soma::LossType {
	const std::string MSE = "mse";
	const std::string CCE = "cce";
}

#endif
