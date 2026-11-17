#ifndef SOMA_LAYER_DENSE_HPP
#define SOMA_LAYER_DENSE_HPP

#include <cstdint>
#include <fstream>
#include <memory>

#include "./layer.hpp"
#include "../matrix.hpp"

namespace Soma::Layers {
	class DenseLayer : public Layer {
	public:
		Matrix W, B;
		DenseLayer(int in, int out) : Layer(in, out) {
			W.resize(out, in);
			B.resize(out, 1);
			params = { std::ref(W), std::ref(B) };
			grads.resize(2);
			grads[0].resize(out, in);
			grads[1].resize(out, 1);
		}

		Matrix run(const Matrix& x) override { return Matrix::dot(W, x) + B; } // y = W . x + B
		Matrix forward(const Matrix& x) override { lastInput = x; return run(x); }
		Matrix backward(const Matrix& dy) override {
			grads[0] = Matrix::dot(dy, lastInput.transpose()); // dW = dy . x^T
			grads[1] = dy; // dB = dy
			return Matrix::dot(W.transpose(), dy); // dx = W^T . dy
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
			for (Matrix& param : params) {
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
			for (Matrix& mat : layer->params) {
				for (int i = 0; i < mat.rows(); i++) {
					in.read(reinterpret_cast<char*>(mat[i].data()), mat.cols() * sizeof(double));
				}
			}
			return layer;
		}
	};
}

#endif
