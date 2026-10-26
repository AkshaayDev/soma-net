#include "../../soma.hpp"
#include <fstream>

// In this example, we will be using a simple 2,2,1 architecture to approximate an XOR gate
int main() {
	NeuralNetwork nn;
	// Use sigmoid activations
	nn.addLayer<DenseLayer>(2, 2);
	nn.addLayer<ActivationLayer>(2, ActivationType::Sigmoid);
	nn.addLayer<DenseLayer>(2, 1);
	nn.addLayer<ActivationLayer>(1, ActivationType::Sigmoid);
	// Initialize weights using the Xavier Normal initialization
	Initialization::xavierNormal(nn);
	// Use Mean Squared Error loss function
	nn.setLossFunction(LossType::MSE);
	// Set training data for the network
	std::vector<std::pair<NNMatrix, NNMatrix>> data = {
		{ NNMatrix::fromVector({0,0}), NNMatrix::fromScalar(0.0) }, // 0 ^ 0 = 0
		{ NNMatrix::fromVector({0,1}), NNMatrix::fromScalar(1.0) }, // 0 ^ 1 = 1
		{ NNMatrix::fromVector({1,0}), NNMatrix::fromScalar(1.0) }, // 1 ^ 0 = 1
		{ NNMatrix::fromVector({1,1}), NNMatrix::fromScalar(0.0) }  // 1 ^ 1 = 0
	};
	// Train the network with the training data with gradient descent
	GradientDescentOptimizer gd(nn, 1); // learningRate = 1
	Trainer trainer(nn, gd, data);
	trainer.enableShuffling = false;
	trainer.train(10000);
	// Test the network by running each test data
	for (std::pair<NNMatrix, NNMatrix> sample : data) {
		std::cout << sample.first[0][0];
		std::cout << " ^ ";
		std::cout << sample.first[1][0];
		std::cout << " = ";
		std::cout << nn.run(sample.first)[0][0] << '\n';
	}
	// Write the neural network data to a file `./out.dat` with binary encoding
	std::ofstream out("./out.dat", std::ios::binary);
	nn.save(out);
	out.close();
}

/*
On average, the output should look something like:
0 ^ 0 = 0.0187684
0 ^ 1 = 0.981912
1 ^ 0 = 0.977819
1 ^ 1 = 0.0178517
However, sometimes the training is not good enough when there are bad initializations
When this happens, the network struggles to learn the data and may output something like:
0 ^ 0 = 0.334147
0 ^ 1 = 0.332994
1 ^ 0 = 0.974527
1 ^ 1 = 0.333746
*/
