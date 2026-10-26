#ifndef SOMA_TRAINER_HPP
#define SOMA_TRAINER_HPP

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "network.hpp"
#include "optimizer.hpp"

class Trainer {
public:
	NeuralNetwork& nn;
	Optimizer& opt;
	std::vector<std::pair<NNMatrix, NNMatrix>>& batch;
	std::function<void()> iterationCallback = []() {};
	std::function<void()> epochCallback = []() {};
	// Network reference and batch reference constructor
	Trainer(NeuralNetwork& nn, Optimizer& opt, std::vector<std::pair<NNMatrix, NNMatrix>>& batch) : nn(nn), opt(opt), batch(batch) {}
	// Training data is split into smaller samples of `sampleSize` to be processed individually
	// sampleSize is -1 by default, meaning the whole batch is processed at once
	int sampleSize = -1;
	// Training data is shuffled before every epoch by default
	bool enableShuffling = true;

	// Train the network
	void train(int epochs, unsigned int shuffleSeed = std::chrono::system_clock::now().time_since_epoch().count()) {
		std::mt19937 gen(shuffleSeed);
		int actualSize = (sampleSize == -1) ? batch.size() : sampleSize;

		for (int epoch = 1; epoch <= epochs; epoch++) {
			if (enableShuffling) std::shuffle(batch.begin(), batch.end(), gen);
			for (int i = 0; i < batch.size(); i += actualSize) {
				std::vector<std::pair<NNMatrix, NNMatrix>> sample(
					batch.begin() + i,
					batch.begin() + std::min(i + actualSize, static_cast<int>(batch.size()))
				);
				nn.averageGrads(sample);
				opt.update();
				nn.iterationsTrained++;
				iterationCallback();
			}
			nn.epochsTrained++;
			epochCallback();
		}
	}
};

// Trainers are standalone objects and not network attributes
// Therefore, the trainer object being used need not be specified in the network

#endif
