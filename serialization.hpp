#ifndef SOMA_SERIALIZATION_HPP
#define SOMA_SERIALIZATION_HPP

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "network.hpp"
#include "optimizer.hpp"

inline void Soma::NeuralNetwork::save(std::ofstream& out, Optimizers::Optimizer* opt) {
	// Write the signature
	std::uint32_t sig = SIGNATURE;
	out.write(reinterpret_cast<const char*>(&sig), sizeof(std::uint32_t));
	// Write the depth
	out.write(reinterpret_cast<const char*>(&depth), sizeof(int));
	// Write the layers
	for (int i = 0; i < depth; i++) {
		layers[i]->save(out);
	}
	// Write the loss function
	std::uint32_t size = lossFnName.size();
	out.write(reinterpret_cast<const char*>(&size), sizeof(std::uint32_t));
	out.write(lossFnName.c_str(), size);
	// Write the iterations and epochs trained
	out.write(reinterpret_cast<const char*>(&iterationsTrained), sizeof(int));
	out.write(reinterpret_cast<const char*>(&epochsTrained), sizeof(int));
	// Write the optimizer data
	bool includeOptData = (opt != nullptr);
	out.write(reinterpret_cast<const char*>(&includeOptData), sizeof(bool));
	if (includeOptData) {
		// Note: This assumes the number for each OptimizerType is 0-255
		std::uint8_t optType = static_cast<std::uint8_t>(opt->getType());
		out.write(reinterpret_cast<const char*>(&optType), sizeof(std::uint8_t));
		opt->save(out);
	}
}

inline void Soma::NeuralNetwork::load(std::ifstream& in, std::unique_ptr<Optimizers::Optimizer> opt) {
	// Read the signature
	std::uint32_t sig;
	in.read(reinterpret_cast<char*>(&sig), sizeof(std::uint32_t));
	if (sig != SIGNATURE) throw std::runtime_error("Invalid signature read.");
	layers.clear();
	avgGrads.clear();
	// Read the depth
	in.read(reinterpret_cast<char*>(&depth), sizeof(int));
	// Read the layers
	for (int i = 0; i < depth; i++) {
		std::unique_ptr<Layers::Layer> layer = Layers::Layer::load(in);
		layers.emplace_back(std::move(layer));
		avgGrads.push_back(layers.back()->grads);
	}
	// Read the loss function
	std::uint32_t size = 0;
	in.read(reinterpret_cast<char*>(&size), sizeof(std::uint32_t));
	lossFnName.resize(size);
	in.read(&lossFnName[0], size);
	setLossFunction(lossFnName);
	// Read the iterations and epochs trained
	in.read(reinterpret_cast<char*>(&iterationsTrained), sizeof(int));
	in.read(reinterpret_cast<char*>(&epochsTrained), sizeof(int));
	// Read the optimizer data
	bool includeOptData;
	in.read(reinterpret_cast<char*>(&includeOptData), sizeof(bool));
	if (includeOptData) {
		// Note: This assumes the number for each OptimizerType is 0-255
		std::uint8_t optType;
		in.read(reinterpret_cast<char*>(&optType), sizeof(std::uint8_t));
		opt = Optimizers::Optimizer::getByType(static_cast<OptimizerType>(optType), *this);
		opt->load(in);
	}
}

#endif
