#ifndef SOMA_LAYER_SIREN_HPP
#define SOMA_LAYER_SIREN_HPP

#include <cstdint>
#include <fstream>
#include <memory>

#include "./layer.hpp"
#include "../matrix.hpp"

namespace Soma::Layers {
	class SIRENLayer : public Layer {
	public:
		Matrix W, B, lastZ;
		double omega0 = 1.0;
		SIRENLayer(int in, int out) : Layer(in, out) {
			W.resize(out, in);
			B.resize(out, 1);
			params = { std::ref(W), std::ref(B) };
			grads.resize(2);
			grads[0].resize(out, in);
			grads[1].resize(out, 1);
		}

		Matrix run(const Matrix& x) override {
			Matrix z = Matrix::dot(W, x) + B; // z = W . x + B
			z.forEach([this](double *val, int, int) {
				*val = std::sin(omega0 * *val); // y = sin(omega0 * z)
			});
			return z;
		}
		Matrix forward(const Matrix& x) override {
			lastInput = x;
			Matrix z = Matrix::dot(W, x) + B; // z = W . x + B
			lastZ = z;
			z.forEach([this](double *val, int, int) {
				*val = std::sin(omega0 * *val); // y = sin(omega0 * z)
			});
			return z;
		}
		Matrix backward(const Matrix& dy) override {
			Matrix dz = lastZ; // dz = dy * omega0 cos(omega0 * z)
			dz.forEach([this, &dy](double *val, int i, int j) {
				*val = dy[i][j] * omega0 * std::cos(omega0 * *val);
			});
			grads[0] = Matrix::dot(dz, lastInput.transpose()); // dW = dz . x^T
			grads[1] = dz; // dB = dz
			return Matrix::dot(W.transpose(), dz); // dx = W^T . dz
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
			for (Matrix& param : params) {
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
