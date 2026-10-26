#ifndef SOMA_LAYER_HPP
#define SOMA_LAYER_HPP

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "activation.hpp"
#include "matrix.hpp"

class Layer {
public:
	int inCount = 0, outCount = 0; // Number of input and output neurons for the layer

	// Optional parameters and gradients for training
	std::vector<std::reference_wrapper<NNMatrix>> params;
	std::vector<NNMatrix> grads;
	// Optional last input and output storage for backpropagation
	NNMatrix lastInput, lastOutput;

	Layer(int inCount = 0, int outCount = 0) : inCount(inCount), outCount(outCount) {}

	virtual ~Layer() = default;
	// Returns an output without setting last input or output
	virtual NNMatrix run(const NNMatrix& x) = 0;
	// Returns an output and sets last input and/or output
	virtual NNMatrix forward(const NNMatrix& x) = 0;
	// Sets gradients and returns error for input
	virtual NNMatrix backward(const NNMatrix& dy) = 0;

	// Save layer data to the file stream
	virtual void save(std::ofstream& out) = 0;
	// Factory loader
	static std::unique_ptr<Layer> load(std::ifstream& in);
};

class ActivationLayer : public Layer {
public:
	std::string fnName;
	std::function<NNMatrix(NNMatrix)> f, g;
	ActivationLayer(int count, std::string fnName) : Layer(count, count), fnName(fnName) {
		if (fnName == ActivationType::Sigmoid) {
			f = Activation::sigmoid;
			g = [this](NNMatrix dy) { return Activation::sigmoidDerivative(lastOutput) * dy; };
		} else if (fnName == ActivationType::ReLU) {
			f = Activation::relu;
			g = [this](NNMatrix dy) { return Activation::reluDerivative(lastOutput) * dy; };
		} else if (fnName == ActivationType::Tanh) {
			f = Activation::tanh;
			g = [this](NNMatrix dy) { return Activation::tanhDerivative(lastOutput) * dy; };		
		} else if (fnName == ActivationType::Softmax) {
			f = Activation::softmax;
			g = [this](NNMatrix dy) { return Activation::softmaxDerivative(lastOutput, dy); };
		} else throw std::runtime_error("Unknown hidden activation function ('" + fnName + "')");
	}

	NNMatrix run(const NNMatrix& x) override { return f(x); }
	NNMatrix forward(const NNMatrix& x) override { lastOutput = f(x); return lastOutput; }
	NNMatrix backward(const NNMatrix& dy) override { return g(dy); }

	void save(std::ofstream& out) override {
		// Write the layer type
		const std::string type = "Activation";
		std::uint32_t size = type.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(std::uint32_t));
		out.write(type.c_str(), size);
		// Write the number of neurons
		out.write(reinterpret_cast<const char*>(&inCount), sizeof(int));
		// Write the activation function name
		size = fnName.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(std::uint32_t));
		out.write(fnName.c_str(), size);
	}
	static std::unique_ptr<ActivationLayer> load(std::ifstream& in) {
		// Layer type was read by static Layer::load
		// Read the number of neurons
		int count;
		in.read(reinterpret_cast<char*>(&count), sizeof(int));
		// Read the activation function
		std::uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(std::uint32_t));
		std::string fnName;
		fnName.resize(size);
		in.read(&fnName[0], size);
		return std::make_unique<ActivationLayer>(count, fnName);
	}
};

class DenseLayer : public Layer {
public:
	NNMatrix W, B;
	DenseLayer(int in, int out) : Layer(in, out) {
		W.resize(out, in);
		B.resize(out, 1);
		params = { std::ref(W), std::ref(B) };
		grads.resize(2);
		grads[0].resize(out, in);
		grads[1].resize(out, 1);
	}

	NNMatrix run(const NNMatrix& x) override { return NNMatrix::dot(W, x) + B; } // y = W . x + B
	NNMatrix forward(const NNMatrix& x) override { lastInput = x; return run(x); }
	NNMatrix backward(const NNMatrix& dy) override {
		grads[0] = NNMatrix::dot(dy, lastInput.transpose()); // dW = dy . x^T
		grads[1] = dy; // dB = dy
		return NNMatrix::dot(W.transpose(), dy); // dx = W^T . dy
	}

	void save(std::ofstream& out) override {
		// Write the layer type
		const std::string type = "Dense";
		std::uint32_t size = type.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(std::uint32_t));
		out.write(type.c_str(), size);
		// Write the number of input and output neurons
		out.write(reinterpret_cast<const char*>(&inCount), sizeof(int));
		out.write(reinterpret_cast<const char*>(&outCount), sizeof(int));
		// Write the weights and biases
		for (NNMatrix& param : params) {
			param.forEach([&out](double *val, int, int) {
				out.write(reinterpret_cast<const char*>(val), sizeof(double));
			});
		}
	}
	static std::unique_ptr<DenseLayer> load(std::ifstream& in) {
		// Layer type was read by static Layer::load
		// Read the number of input and output neurons
		int inCount, outCount;
		in.read(reinterpret_cast<char*>(&inCount), sizeof(int));
		in.read(reinterpret_cast<char*>(&outCount), sizeof(int));
		std::unique_ptr<DenseLayer> layer = std::make_unique<DenseLayer>(inCount, outCount);
		// Read the weights and biases
		for (NNMatrix& mat : layer->params) {
			for (int i = 0; i < mat.rows(); i++) {
				in.read(reinterpret_cast<char*>(mat[i].data()), mat.cols() * sizeof(double));
			}
		}
		return layer;
	}
};

class SIRENLayer : public Layer {
public:
	NNMatrix W, B, lastZ;
	double omega0 = 1.0;
	SIRENLayer(int in, int out) : Layer(in, out) {
		W.resize(out, in);
		B.resize(out, 1);
		params = { std::ref(W), std::ref(B) };
		grads.resize(2);
		grads[0].resize(out, in);
		grads[1].resize(out, 1);
	}

	NNMatrix run(const NNMatrix& x) override {
		NNMatrix z = NNMatrix::dot(W, x) + B; // z = W . x + B
		z.forEach([this](double *val, int, int) {
			*val = std::sin(omega0 * *val); // y = sin(omega0 * z)
		});
		return z;
	}
	NNMatrix forward(const NNMatrix& x) override {
		lastInput = x;
		NNMatrix z = NNMatrix::dot(W, x) + B; // z = W . x + B
		lastZ = z;
		z.forEach([this](double *val, int, int) {
			*val = std::sin(omega0 * *val); // y = sin(omega0 * z)
		});
		return z;
	}
	NNMatrix backward(const NNMatrix& dy) override {
		NNMatrix dz = lastZ; // dz = dy * omega0 cos(omega0 * z)
		dz.forEach([this, &dy](double *val, int i, int j) {
			*val = dy[i][j] * omega0 * std::cos(omega0 * *val);
		});
		grads[0] = NNMatrix::dot(dz, lastInput.transpose()); // dW = dz . x^T
		grads[1] = dz; // dB = dz
		return NNMatrix::dot(W.transpose(), dz); // dx = W^T . dz
	}

	void save(std::ofstream& out) override {
		// Write the layer type
		const std::string type = "SIREN";
		std::uint32_t size = type.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(std::uint32_t));
		out.write(type.c_str(), size);
		// Write the number of input and output neurons
		out.write(reinterpret_cast<const char*>(&inCount), sizeof(int));
		out.write(reinterpret_cast<const char*>(&outCount), sizeof(int));
		// Write omega0
		out.write(reinterpret_cast<const char*>(&omega0), sizeof(double));
		// Write the weights and biases
		for (NNMatrix& param : params) {
			param.forEach([&out](double *val, int, int) {
				out.write(reinterpret_cast<const char*>(val), sizeof(double));
			});
		}
	}
	static std::unique_ptr<SIRENLayer> load(std::ifstream& in) {
		// Layer type was read by static Layer::load
		// Read the number of input and output neurons
		int inCount, outCount;
		in.read(reinterpret_cast<char*>(&inCount), sizeof(int));
		in.read(reinterpret_cast<char*>(&outCount), sizeof(int));
		std::unique_ptr<SIRENLayer> layer = std::make_unique<SIRENLayer>(inCount, outCount);
		// Read omega0
		in.read(reinterpret_cast<char*>(&layer->omega0), sizeof(double));
		// Read the weights and biases
		for (NNMatrix& mat : layer->params) {
			for (int i = 0; i < mat.rows(); i++) {
				in.read(reinterpret_cast<char*>(mat[i].data()), mat.cols() * sizeof(double));
			}
		}
		return layer;
	}
};

inline std::unique_ptr<Layer> Layer::load(std::ifstream& in) {
	std::string type;
	std::uint32_t size = 0;
	in.read(reinterpret_cast<char*>(&size), sizeof(std::uint32_t));
	type.resize(size);
	in.read(&type[0], size);
	if (type == "Activation") return ActivationLayer::load(in);
	if (type == "Dense") return DenseLayer::load(in);
	if (type == "SIREN") return SIRENLayer::load(in);
	throw std::runtime_error("Unknown layer type found.");
}

#endif
