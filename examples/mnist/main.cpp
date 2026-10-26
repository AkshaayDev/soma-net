#include "../../soma.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h> // Multithreading to speed up testing dataset loss calculation

// Returns the dataset of from `imgPath` and `lblPath`
// Adapted from https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
std::vector<std::pair<NNMatrix, NNMatrix>> loadMNIST(std::string imgPath, std::string lblPath) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};
	std::ifstream images(imgPath, std::ios::binary);
	std::ifstream labels(lblPath, std::ios::binary);
	if (!images.is_open()) throw std::runtime_error("Cannot open image dataset file");
	if (!labels.is_open()) throw std::runtime_error("Cannot open label dataset file");

	int magicNumber = 0;
	int totalImages = 0, rows = 0, cols = 0;
	// Read image magic number
	images.read((char*)&magicNumber, sizeof(magicNumber)); magicNumber = reverseInt(magicNumber);
	if(magicNumber != 2051) throw std::runtime_error("Invalid MNIST image file!");
	// Read total number of images, rows and columns
	images.read((char*)&totalImages, sizeof(totalImages)), totalImages = reverseInt(totalImages);
	images.read((char*)&rows, sizeof(rows)), rows = reverseInt(rows);
	images.read((char*)&cols, sizeof(cols)), cols = reverseInt(cols);
	// Read label magic number
	labels.read((char*)&magicNumber, sizeof(magicNumber)); magicNumber = reverseInt(magicNumber);
	if (magicNumber != 2049) throw std::runtime_error("Invalid MNIST label file!");
	// Read total number of labels
	int totalLabels = 0;
	labels.read((char*)&totalLabels, sizeof(totalLabels)), totalLabels = reverseInt(totalLabels);

	if (totalImages != totalLabels) {
		throw std::runtime_error(std::to_string(totalImages) + " images found but " + std::to_string(totalLabels) + " labels found.");
	}

	std::vector<std::pair<NNMatrix, NNMatrix>> dataset(totalImages);
	for (int i = 0; i < totalImages; i++) {
		// Form a label column matrix
		unsigned char label;
		labels.read((char*)&label, sizeof(label));
		std::vector<double> expected(10, 0);
		expected[static_cast<int>(label)] = 1;
		std::pair<NNMatrix, NNMatrix> pair(NNMatrix(rows*cols, 1), NNMatrix::fromVector(expected));
		// Read normalized pixel grayscale data
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				unsigned char pixel = 0;
				images.read((char*)&pixel, sizeof(pixel));
				pair.first[r*cols+c][0] = static_cast<double>(pixel) / 255.0;
			}
		}
		dataset[i] = pair;
	}
	return dataset;
}

NeuralNetwork nn;
std::vector<std::pair<NNMatrix, NNMatrix>> trainset, testset;
AdamOptimizer adam;

// Calculate average loss of the test set (Uses multithreading if OpenMP is used)
double avgLoss() {
	double totalLoss = 0.0;
	#pragma omp parallel for reduction(+:totalLoss)
	for (int i = 0; i < testset.size(); i++) {
		totalLoss += nn.lossFn(nn.run(testset[i].first), testset[i].second);
	}
	return totalLoss / testset.size();
}
// Log iteration number every iteration and average loss of the test set every 20 iterations
void iterationCallback() {
	std::cout << "Iteration " << nn.iterationsTrained;
	if (nn.iterationsTrained % 20 == 0) std::cout << ", Avg Loss: " << avgLoss();
	std::cout << '\n';
}
// Log epoch number and write network data to `./nn.dat`
void epochCallback() {
	std::cout << "Epoch " << nn.epochsTrained << " finished.\n";
	std::ofstream out("./nn.dat", std::ios::binary);
	nn.save(out, &adam);
	out.close();
}

// In this example, we will train a neural network to recognize images of handwritten digits.
// Its input is a flattened 28x28 column matrix of grayscale values between 0 and 1
// Its output is a confidence column matrix for each digit 0-9
// MNIST dataset downloaded from https://drive.google.com/file/d/11ZiNnV3YtpZ7d9afHZg0rtDRrmhha-1E/view
// MNIST dataset info: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
// The files should be extracted and placed in a `./data` folder
// The trained network can be tested with `nnpaint.cpp`.
// Important: Compile with OpenMP to use multithreading. The highest level of compiler optimization is recommended.
// Example compilation command: `g++ main.cpp -O3 -fopenmp`
int main() {
	// Load training data and initialize the network
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			std::cout << "Loading training images\n";
			trainset = loadMNIST("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte");
			std::cout << "Training images loaded\n";
		}
		#pragma omp section
		{
			std::cout << "Loading testing images\n";
			testset = loadMNIST("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte");
			std::cout << "Testing images loaded\n";
		}
	}
	nn.addLayer<DenseLayer>(784, 128);
	nn.addLayer<ActivationLayer>(128, ActivationType::ReLU);
	nn.addLayer<DenseLayer>(128, 64);
	nn.addLayer<ActivationLayer>(64, ActivationType::ReLU);
	nn.addLayer<DenseLayer>(64, 10);
	nn.addLayer<ActivationLayer>(128, ActivationType::Softmax);
	nn.setLossFunction(LossType::CCE);
	Initialization::heNormal(nn);
	adam.init(nn);
	
	// If there exists a data file `./nn.dat`, read from it
	std::ifstream in("./nn.dat", std::ios::binary);
	if (in.good()) nn.load(in, std::make_unique<AdamOptimizer>(adam));
	in.close();

	std::cout << "Training starting after " << nn.epochsTrained << " epochs and " << nn.iterationsTrained << " iterations.\n";
	std::cout << "Current average testset loss: " << avgLoss() << '\n';
	// The loss is usually around log_e(1/10) or ~2.30 after initialization
	Trainer trainer(nn, adam, trainset);
	trainer.sampleSize = 128;
	trainer.iterationCallback = iterationCallback;
	trainer.epochCallback = epochCallback;
	trainer.train(30);
	std::cout << "Training finished." << std::endl;
}
