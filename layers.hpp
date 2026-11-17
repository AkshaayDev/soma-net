#ifndef SOMA_LAYERS_HPP
#define SOMA_LAYERS_HPP

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "./layer/layer.hpp"
#include "./layer/dense.hpp"
#include "./layer/activation.hpp"
#include "./layer/siren.hpp"

namespace Soma::Layers {
	inline std::unique_ptr<Layer> Soma::Layers::Layer::load(std::ifstream& in) {
		std::string type;
		std::uint32_t size = 0;
		in.read(reinterpret_cast<char*>(&size), sizeof(std::uint32_t));
		type.resize(size);
		in.read(&type[0], size);
		if (type == "Dense") return DenseLayer::load(in);
		if (type == "Activation") return ActivationLayer::load(in);
		if (type == "SIREN") return SIRENLayer::load(in);
		throw std::runtime_error("Unknown layer type found.");
	}
}

#endif
