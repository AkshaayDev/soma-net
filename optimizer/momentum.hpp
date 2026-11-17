#ifndef SOMA_OPTIMIZER_MOMENTUM_HPP
#define SOMA_OPTIMIZER_MOMENTUM_HPP

#include <fstream>
#include <vector>

#include "../matrix.hpp"
#include "../network.hpp"
#include "./optimizer.hpp"

namespace Soma::Optimizers {
	class MomentumOptimizer : public Optimizer {
	public:
		double learningRate = 0.001;
		double beta = 0.9;
		std::vector<std::vector<Matrix>> velocity;

		MomentumOptimizer() = default;
		MomentumOptimizer(Network& nn) { init(nn); }
		MomentumOptimizer(Network& nn, double learningRate, double beta) : learningRate(learningRate), beta(beta) { init(nn); }
		void init(Network& network) override {
			Optimizer::init(network);
			initMoment(velocity);
		}

		virtual void update() override {
			// v = β * v + (1 - β) * ∂L/∂θ
			// θ = θ - α * v
			for (int i = 0; i < nn->depth; i++) {
				for (int j = 0; j < nn->layers[i]->params.size(); j++) {
					Matrix& param = nn->layers[i]->params[j];
					Matrix& avgGrad = nn->avgGrads[i][j];
					Matrix& v = velocity[i][j];
					v = beta * v + (1 - beta) * avgGrad;
					param = param - learningRate * v;
				}
			}
		};
		virtual OptimizerType getType() const override { return OptimizerType::Momentum; }
		virtual void save(std::ofstream& out) override {
			out.write(reinterpret_cast<const char*>(&learningRate), sizeof(double));
			out.write(reinterpret_cast<const char*>(&beta), sizeof(double));
			saveMoment(velocity, out);
		}
		virtual void load(std::ifstream& in) override {
			in.read(reinterpret_cast<char*>(&learningRate), sizeof(double));
			in.read(reinterpret_cast<char*>(&beta), sizeof(double));
			loadMoment(velocity, in);
		}
	};
}

#endif
