#ifndef SOMA_OPTIMIZER_GD_HPP
#define SOMA_OPTIMIZER_GD_HPP

#include <fstream>

#include "../matrix.hpp"
#include "../network.hpp"
#include "./optimizer.hpp"

namespace Soma::Optimizers {
	class GradientDescentOptimizer : public Optimizer {
	public:
		double learningRate = 0.001;

		GradientDescentOptimizer() = default;
		GradientDescentOptimizer(Network& nn) { init(nn); }
		GradientDescentOptimizer(Network& nn, double learningRate) : learningRate(learningRate) { init(nn); }
		void init(Network& network) override { Optimizer::init(network); }
		
		virtual void update() override {
			// θ = θ - α * ∂L/∂θ
			for (int i = 0; i < nn->depth; i++) {
				for (int j = 0; j < nn->layers[i]->params.size(); j++) {
					Matrix& param = nn->layers[i]->params[j];
					Matrix& avgGrad = nn->avgGrads[i][j];
					param = param - learningRate * avgGrad;
				}
			}
		}
		OptimizerType getType() const override { return OptimizerType::GradientDescent; }
		virtual void save(std::ofstream& out) override {
			out.write(reinterpret_cast<const char*>(&learningRate), sizeof(double));
		}
		virtual void load(std::ifstream& in) override {
			in.read(reinterpret_cast<char*>(&learningRate), sizeof(double));
		}
	};
}

#endif
