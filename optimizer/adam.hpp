#ifndef SOMA_OPTIMIZER_ADAM_HPP
#define SOMA_OPTIMIZER_ADAM_HPP

#include <fstream>
#include <vector>
#include <cmath>

#include "../matrix.hpp"
#include "../network.hpp"
#include "./optimizer.hpp"

namespace Soma::Optimizers {
	class AdamOptimizer : public Optimizer {
	public:
		double learningRate = 0.001;
		double beta1 = 0.9;
		double beta2 = 0.999;
		double epsilon = 1e-8;
		std::vector<std::vector<Matrix>> first, second;

		AdamOptimizer() = default;
		AdamOptimizer(Network& nn) { init(nn); }
		AdamOptimizer(Network& nn, double learningRate, double beta1, double beta2, double epsilon) : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon) { init(nn); }
		void init(Network& network) override {
			Optimizer::init(network);
			initMoment(first);
			initMoment(second);
		}

		virtual void update() override {
			// m = β1 * m + (1 - β1) * ∂L/∂θ
			// v = β2 * v + (1 - β2) * (∂L/∂θ)^2
			// m̂ = m / (1 - (β1)^t)
			// v̂ = v / (1 - (β2)^t)
			// θ = θ - α * m̂/(sqrt(v̂) + ε)
			// Correction coeffecients
			double c1 = 1 - std::pow(beta1, nn->iterationsTrained + 1);
			double c2 = 1 - std::pow(beta2, nn->iterationsTrained + 1);
			for (int i = 0; i < nn->depth; i++) {
				for (int j = 0; j < nn->layers[i]->params.size(); j++) {
					Matrix& param = nn->layers[i]->params[j];
					Matrix& avgGrad = nn->avgGrads[i][j];
					Matrix& m = first[i][j];
					Matrix& v = second[i][j];
					m = beta1 * m + (1 - beta1) * avgGrad;
					v = beta2 * v + (1 - beta2) * (avgGrad ^ 2);
					param = param - learningRate * (m/c1) / (((v/c2) ^ 0.5) + epsilon);
				}
			}
		};
		virtual OptimizerType getType() const override { return OptimizerType::Adam; }
		virtual void save(std::ofstream& out) override {
			out.write(reinterpret_cast<const char*>(&learningRate), sizeof(double));
			out.write(reinterpret_cast<const char*>(&beta1), sizeof(double));
			out.write(reinterpret_cast<const char*>(&beta2), sizeof(double));
			out.write(reinterpret_cast<const char*>(&epsilon), sizeof(double));
			saveMoment(first, out); saveMoment(second, out);
		}
		virtual void load(std::ifstream& in) override {
			in.read(reinterpret_cast<char*>(&learningRate), sizeof(double));
			in.read(reinterpret_cast<char*>(&beta1), sizeof(double));
			in.read(reinterpret_cast<char*>(&beta2), sizeof(double));
			in.read(reinterpret_cast<char*>(&epsilon), sizeof(double));
			loadMoment(first, in); loadMoment(second, in);
		}
	};
}

#endif
