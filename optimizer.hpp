#ifndef SOMA_OPTIMIZER_HPP
#define SOMA_OPTIMIZER_HPP

#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "network.hpp"

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

// Note: All Optimizer update functions assume that the average partial derivatives are already set
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

// Optimizers are standalone objects and not network attributes
// Therefore, the optimizer object being used need not be specified in the network

#endif
