#ifndef SOMA_LAYER_ACTIVATION_HPP
#define SOMA_LAYER_ACTIVATION_HPP

#include <cstdint>
#include <fstream>
#include <memory>

#include "./layer.hpp"
#include "./activationfn.hpp"
#include "../matrix.hpp"

namespace Soma::Layers {
	class ActivationLayer : public Layer {
	public:
		std::string fnName;
		std::function<Matrix(Matrix)> f, g;
		ActivationLayer(int count, std::string fnName) : Layer(count, count), fnName(fnName) {
			if (fnName == ActivationType::Sigmoid) {
				f = Activation::sigmoid;
				g = [this](Matrix dy) { return Activation::sigmoidDerivative(lastOutput) * dy; };
			} else if (fnName == ActivationType::ReLU) {
				f = Activation::relu;
				g = [this](Matrix dy) { return Activation::reluDerivative(lastOutput) * dy; };
			} else if (fnName == ActivationType::Tanh) {
				f = Activation::tanh;
				g = [this](Matrix dy) { return Activation::tanhDerivative(lastOutput) * dy; };		
			} else if (fnName == ActivationType::Softmax) {
				f = Activation::softmax;
				g = [this](Matrix dy) { return Activation::softmaxDerivative(lastOutput, dy); };
			} else throw std::runtime_error("Unknown hidden activation function ('" + fnName + "')");
		}

		Matrix run(const Matrix& x) override { return f(x); }
		Matrix forward(const Matrix& x) override { lastOutput = f(x); return lastOutput; }
		Matrix backward(const Matrix& dy) override { return g(dy); }

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
}

#endif
