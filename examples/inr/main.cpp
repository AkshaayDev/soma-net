#include "../../soma.hpp"
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <omp.h>

NeuralNetwork nn;
const char* imgLoadPath = "./img/img.png";
const char* imgCreatePath = "./res.png";
int width, height;
const int outWidth = 512, outHeight = 512;

// Create the neural representation of the image and save it into `imgCreatePath`
void createImage() {
	unsigned char* data = new unsigned char[outWidth * outHeight * 3];
	#pragma omp parallel for collapse(2) // Parallelize each iteration
	for (int i = 0; i < outHeight; i++) {
		for (int j = 0; j < outWidth; j++) {
			int idx = i * outWidth + j;
			double y = static_cast<double>(i) / (outHeight - 1) * 2 - 1;
			double x = static_cast<double>(j) / (outWidth - 1) * 2 - 1;
			NNMatrix rgb = nn.run(NNMatrix::fromVector({x, y}));
			for (int c = 0; c < 3; c++) {
				// Normalize and clamp from (-1, 1) to (0, 255)
				double pixel = (rgb[c][0] + 1) * 127.5;
				data[3 * idx + c] = std::max(0.0, std::min(255.0, pixel));
			}
		}
	}
	stbi_write_png(imgCreatePath, outWidth, outHeight, 3, data, outWidth * 3);
	delete[] data;
	data = nullptr;
}

std::vector<std::pair<NNMatrix, NNMatrix>> batch;

// Load a the image from `imgLoadPath` and create training batch data
void loadImage() {
	// Load the image data and metadata such as width and height (0 ignores the number of channels)
	unsigned char* data = stbi_load(imgLoadPath, &width, &height, 0, 3); // 3 channels expected (RGB)
	if (data == NULL) {
		throw std::runtime_error("Error loading image: " + std::string(stbi_failure_reason()));
	}
	batch.resize(height * width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int idx = i * width + j;
			std::pair<NNMatrix, NNMatrix> sample;
			double y = static_cast<double>(i) / (height - 1) * 2 - 1;
			double x = static_cast<double>(j) / (width - 1) * 2 - 1;
			sample.first = NNMatrix::fromVector({x, y});
			sample.second.resize(3,1);
			for (int c = 0; c < 3; c++) {
				// Normalize from (0, 255) to (-1, -1)
				sample.second[c][0] = (static_cast<double>(data[3 * idx + c]) / 127.5) - 1;
			}
			batch[idx] = sample;
		}
	}
	stbi_image_free(data);
}

// In this example, we will try to get the network to learn an image with an Implicit Neural Representation(INR)
// To do this, the image is first squished into a square with coordinates and RGB values normalised to (-1, 1)
// Then, it approximates a function f(x,y) => {r,g,b}
// This script loads the image data, trains the network and creates a resulting image approximation
int main() {
	nn.addLayer<SIRENLayer>(2, 32);
	nn.addLayer<SIRENLayer>(32, 32);
	nn.addLayer<DenseLayer>(32, 3);
	Initialization::SIRENInit(nn);
	Initialization::xavierUniform(nn);
	nn.setLossFunction(LossType::MSE);
	AdamOptimizer adam(nn);

	// If there exists a data file `./nn.dat`, read from it
	std::ifstream in("./nn.dat", std::ios::binary);
	if (in.good()) nn.load(in, std::make_unique<AdamOptimizer>(adam));
	in.close();

	// Train the network with adam
	std::cout << "Loading image..." << std::endl;
	loadImage();
	std::cout << "Image loaded." << std::endl;
	Trainer trainer(nn, adam, batch);
	trainer.epochCallback = []() { std::cout << "Epoch " << nn.epochsTrained << "\n"; };
	trainer.sampleSize = 128;
	trainer.train(100);
	std::cout << "Creating image..." << std::endl;
	createImage();
	std::cout << "Image created" << std::endl;

	// Write network data to `./nn.dat`
	std::ofstream out("./nn.dat", std::ios::binary);
	nn.save(out, &adam);
	out.close();
}
