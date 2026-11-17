#ifndef SOMA_OPTIMIZERS_HPP
#define SOMA_OPTIMIZERS_HPP

#include <memory>
#include <stdexcept>

#include "./network.hpp"
#include "./optimizer/optimizer.hpp"
#include "./optimizer/gd.hpp"
#include "./optimizer/momentum.hpp"
#include "./optimizer/adam.hpp"

namespace Soma::Optimizers {
	inline std::unique_ptr<Optimizer> Optimizer::getByType(OptimizerType type, Network& nn) {
		switch (type) {
			case OptimizerType::GradientDescent:
				return std::make_unique<GradientDescentOptimizer>(nn);
			case OptimizerType::Momentum:
				return std::make_unique<MomentumOptimizer>(nn);
			case OptimizerType::Adam:
				return std::make_unique<AdamOptimizer>(nn);
			default:
				throw std::runtime_error("Unknown optimizer type found");
		}
	}
}

#endif
