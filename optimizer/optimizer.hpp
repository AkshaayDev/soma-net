#ifndef SOMA_OPTIMIZER_OPTIMIZER_HPP
#define SOMA_OPTIMIZER_OPTIMIZER_HPP

#include <fstream>
#include <memory>
#include <vector>

#include "../matrix.hpp"
#include "../network.hpp"

namespace Soma {
	enum class OptimizerType { GradientDescent, Momentum, Adam };
}

namespace Soma::Optimizers {
	class Optimizer {
	public:
		Network* nn = nullptr;

		Optimizer() = default;
 	   virtual ~Optimizer() = default;

		virtual void init(Network& network) { nn = &network; }

		// Optimizer update functions assume that the average gradients are already set
		virtual void update() = 0;
		virtual OptimizerType getType() const = 0;
		virtual void save(std::ofstream& out) = 0;
		virtual void load(std::ifstream& in) = 0;

		static std::unique_ptr<Optimizer> getByType(OptimizerType type, Network& nn);
	protected:
		// Initialize moments (Layers cannot change after attaching optimizer)
		void initMoment(std::vector<std::vector<Matrix>>& moment) {
			moment.resize(nn->depth);
			for (int i = 0; i < nn->depth; i++) {
				moment[i].resize(nn->layers[i]->params.size());
				for (int j = 0; j < nn->layers[i]->params.size(); j++) {
					Matrix& param = nn->layers[i]->params[j];
					moment[i][j].resize(param.rows(), param.cols());
					moment[i][j].fill(0.0);
				}
			}
		}
		// Helper to write moments to an output file stream (Assuming matching dimensions)
		void saveMoment(std::vector<std::vector<Matrix>>& moment, std::ofstream& out) {
			for (std::vector<Matrix>& layerMoment : moment) {
				for (Matrix& gradMoment : layerMoment) {
					gradMoment.forEach([&out](double *val, int, int) {
						out.write(reinterpret_cast<const char*>(val), sizeof(double));
					});
				}
			}
		}
		// Helper to read moments from an input file stream (Assuming matching dimensions)
		void loadMoment(std::vector<std::vector<Matrix>>& moment, std::ifstream& in) {
			for (std::vector<Matrix>& layerMoment : moment) {
				for (Matrix& gradMoment : layerMoment) {
					for (int i = 0; i < gradMoment.rows(); i++) {
						in.read(reinterpret_cast<char*>(gradMoment[i].data()), gradMoment.cols() * sizeof(double));
					}
				}
			}
		}
	};
}

#endif
