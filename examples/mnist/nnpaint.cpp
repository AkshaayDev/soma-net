/*
This is a simple 28x28 grid paint app made with raylib that shows the network's guess of the digit drawn
This script reads the network from a file `nn.dat` and uses it to predict the digit

Usage:
1. Download or install raylib (Official website: https://www.raylib.com/)
2. Ensure that a `nn.dat` file exists in the same directory as this file
3. Compile with raylib and run

Controls:
Left Mouse: White brush
Right Mouse: Black brush/Erase
'C' Key: Clear canvas
*/
#include "../../soma.hpp"
#include "raylib.h"

NNMatrix pixels(28 * 28, 1);
const int pixelScale = 20;
const int brushSize = 2;

static void draw(int delta) {
	int x = GetMouseX() / pixelScale;
	int y = GetMouseY() / pixelScale;
	if (x < 0 || x >= 28 || y < 0 || y >= 28) return;

	for (int i = -brushSize; i < brushSize; i++) {
		for (int j = -brushSize; j < brushSize; j++) {
			int nx = x + i, ny = y + j;
			// Make sure current pixel is in the grid
			if (nx < 0 || nx >= 28 || ny < 0 || ny >= 28) continue;

			// Normalized distance from center
			double d = std::hypot(i, j) / brushSize;
			double intensity = 0;
			if (d < 1) {
				// Apply cos^2(d * pi/2)
				intensity = std::pow(std::cos(d * 3.1415926536 / 2), 2);
			}
			double& pixel = pixels[28 * ny + nx][0];
			// Clamp between 0-1
			pixel = std::max(0.0, std::min(pixel + delta * intensity, 1.0));
		}
	}
}

int main() {
	// Load network data from `./nn.dat`
	NeuralNetwork nn;
	std::ifstream in("./nn.dat", std::ios::binary);
	if (in.good()) nn.load(in);
	else std::cerr << "Could not open network data file\n";
	in.close();

	// Create the window
	InitWindow(28 * pixelScale + 200, 28 * pixelScale, "NNPaint");
	SetTargetFPS(60);
	SetConfigFlags(FLAG_VSYNC_HINT);

	std::vector<Color> pixelColors(28 * 28);
	Image img = GenImageColor(28, 28, BLACK);
	Texture2D texture = LoadTextureFromImage(img);
	UnloadImage(img);

	while (!WindowShouldClose()) {
		// Left Mouse: White brush
		if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) draw(1);
		// Right Mouse: Black brush/Erase
		if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) draw(-1);
		// 'C' Key: Clear canvas
		if (IsKeyPressed(KEY_C))
			for (int i = 0; i < 28 * 28; i++)
				pixels[i][0] = 0;

		// Fill the color map and update the texture using the pixel matrix
		for (int i = 0; i < 28*28; i++) {
			unsigned char intensity = static_cast<unsigned char>(pixels[i][0] * 255);
			pixelColors[i] = { intensity, intensity, intensity, 255 };
		}
		UpdateTexture(texture, pixelColors.data());

		BeginDrawing();
		ClearBackground({ 20, 20, 20, 255 });

		// Draw the grid
		DrawTextureEx(texture, { 0,0 }, 0, pixelScale, WHITE);
		DrawText("Left Mouse: White brush", 10, 10, 20, DARKGRAY);
		DrawText("Right Mouse: Black brush", 10, 30, 20, DARKGRAY);
		DrawText("\'C\' Key: Clear canvas", 10, 50, 20, DARKGRAY);

		// Draw the predictions
		NNMatrix res = nn.run(pixels);
		for (int i = 0; i < 10; i++) {
			int x = 28 * pixelScale + 40;
			int y = 50 + i * 50;
			unsigned char intensity = res[i][0] * 255;
			unsigned char whitened = 128 + intensity / 2;
			Color confidence = { intensity, intensity, intensity, 255 };
			Color contrast = (intensity > 127) ? BLACK : WHITE;
			Color visible = { whitened, whitened, whitened, 255 };
			// Draw a circle with the brightness corresponding to confidence
			DrawCircle(x, y, 20, confidence);
			// Draw its number label and outline with a contrasting color
			DrawCircleLines(x, y, 20, contrast);
			DrawText(std::to_string(i).c_str(), x - 5, y - 10, 20, contrast);
			// Draw its confidence percentage with a more visible color
			std::string percentage = std::to_string(res[i][0] * 100) + "%";
			DrawText(percentage.c_str(), x + 30, y - 10, 20, visible);
		}
		EndDrawing();
	}
	UnloadTexture(texture);
	CloseWindow();
	return 0;
}
